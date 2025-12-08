#!/usr/bin/env python3
"""
EMG Realtime GUI
- Single-file Tkinter + matplotlib GUI wrapper around the user's realtime predictor
- Shows live raw signal plot and live probability trace
- Displays current p_raw, p_used (EMA), votes, and decided label
- Provides controls for COM port, baud, model/scaler/encoder paths, Start/Stop, Threshold, EMA toggle/alpha and Vote window settings

Run:
    python emg_realtime_gui.py

Dependencies:
    pip install pyserial joblib numpy matplotlib

Notes:
- This reuses the feature extraction and model/scaler/encoder loading logic from the realtime script.
- Serial reading and model inference run in a background thread; the GUI receives updates via a thread-safe queue.
- If you prefer PyQt or pyqtgraph (faster plotting), this file can be adapted.
"""

import threading
import queue
import time
from collections import deque
import numpy as np
import joblib
import serial
import sys
import os

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------ Default config (can be edited in GUI) ------------------
DEFAULT_COM = "COM5"
DEFAULT_BAUD = 115200
DEFAULT_MODEL = "emg_mlp_model_clean.joblib"
DEFAULT_SCALER = "emg_scaler_clean.pkl"
DEFAULT_LE = "emg_label_encoder_clean.pkl"

# Window & step identical to your realtime script
WIN_SIZE = 89
STEP_SIZE = 22
ACTIVE_LABEL = "ACTIVE"
# ---------------------------------------------------------------------------

# ---------- Feature extraction / model helpers (copied and slightly adapted) ----------

def extract_emg_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    x_centered = x - np.mean(x)
    x_abs = np.abs(x_centered)
    std = np.std(x_centered)
    thr = 0.05 * std if std > 0 else 0.0

    mav = np.mean(x_abs)
    rms = np.sqrt(np.mean(x_centered ** 2))
    var = np.var(x_centered)
    wl = np.sum(np.abs(np.diff(x_centered)))

    zc = 0
    if x_centered.size > 1:
        zc = np.sum(
            ((x_centered[:-1] * x_centered[1:]) < 0) &
            (np.abs(np.diff(x_centered)) > thr)
        )

    ssc = 0
    if x_centered.size > 2:
        dx1 = np.diff(x_centered)
        ssc = np.sum(
            ((dx1[:-1] * dx1[1:]) < 0) &
            (np.abs(dx1[:-1]) > thr) &
            (np.abs(dx1[1:]) > thr)
        )

    wamp = np.sum(np.abs(np.diff(x_centered)) > thr) if x_centered.size > 1 else 0
    iemg = np.sum(x_abs)

    return np.array([mav, rms, var, wl, zc, ssc, wamp, iemg], dtype=float)


def load_model_scaler_encoder(model_path, scaler_path=None, le_path=None):
    model = None
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_path}': {e}")

    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print("Warning: failed to load scaler:", e)

    le = None
    if le_path and os.path.exists(le_path):
        try:
            le = joblib.load(le_path)
        except Exception as e:
            print("Warning: failed to load label encoder:", e)

    return model, scaler, le

def get_active_probability(model, feats, le, active_label="ACTIVE"):
    # feats must be shape (1, n_features)

    # get probability matrix
    probs = model.predict_proba(feats)   # shape (1, n_classes)

    # determine class order
    if le is not None and hasattr(le, "classes_"):
        classes = [str(c).strip().upper() for c in le.classes_]
    else:
        classes = [str(c).strip().upper() for c in model.classes_]

    # find index of ACTIVE
    try:
        idx = classes.index(active_label.strip().upper())
    except ValueError:
        # fallback: highest probability
        print("[WARN] ACTIVE not in classes:", classes)
        return float(np.max(probs[0]))

    return float(probs[0, idx])


# --------------------------------------------------------------------------------------

class SerialWorker(threading.Thread):
    """Background thread that reads serial, computes features & probabilities and pushes updates to GUI queue."""
    def __init__(self, com, baud, model_path, scaler_path, le_path, gui_queue,
                 win_size=WIN_SIZE, step_size=STEP_SIZE,
                 prob_threshold=0.85, use_ema=True, ema_alpha=0.4,
                 vote_window=4, required_votes=2, idle_sleep=0.001):
        super().__init__(daemon=True)
        self.com = com
        self.baud = int(baud)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.le_path = le_path
        self.gui_queue = gui_queue
        self.win_size = win_size
        self.step_size = step_size
        self.prob_threshold = float(prob_threshold)
        self.use_ema = use_ema
        self.ema_alpha = float(ema_alpha)
        self.vote_window = int(vote_window)
        self.required_votes = int(required_votes)
        self.idle_sleep = float(idle_sleep)

        self._stop_event = threading.Event()

        # runtime
        self.buffer = deque(maxlen=self.win_size)
        self.sample_count = 0
        self.votes = deque(maxlen=self.vote_window)
        self.ema_prob = None

        self.serial = None
        self.model = None
        self.scaler = None
        self.le = None

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        # load model first
        try:
            self.model, self.scaler, self.le = load_model_scaler_encoder(self.model_path,
                                                                          self.scaler_path,
                                                                          self.le_path)
        except Exception as e:
            self.gui_queue.put(("error", f"Model load failed: {e}"))
            return

        # open serial
        try:
            self.serial = serial.Serial(self.com, self.baud, timeout=1)
        except Exception as e:
            self.gui_queue.put(("error", f"Serial open failed: {e}"))
            return

        # allow device to settle
        time.sleep(1.0)
        self.gui_queue.put(("info", f"Connected {self.com}@{self.baud}"))

        try:
            while not self.stopped():
                if self.serial.in_waiting > 0:
                    try:
                        line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    except Exception:
                        continue
                    if not line:
                        continue
                    try:
                        val = float(line)
                    except Exception:
                        continue

                    self.buffer.append(val)
                    self.sample_count += 1

                    # push latest raw sample (for live raw plot)
                    self.gui_queue.put(("sample", val))

                    if len(self.buffer) == self.win_size and (self.sample_count % self.step_size == 0):
                        feats = extract_emg_features(np.array(self.buffer))
                        if feats is None:
                            continue
                        feats_2d = feats.reshape(1, -1)

                        if self.scaler is not None:
                            try:
                                feats_in = self.scaler.transform(feats_2d)
                            except Exception:
                                feats_in = feats_2d
                        else:
                            feats_in = feats_2d

                        p_active = get_active_probability(self.model, feats_in, self.le)


                        if self.use_ema:
                            if self.ema_prob is None:
                                self.ema_prob = p_active
                            else:
                                self.ema_prob = self.ema_alpha * p_active + (1.0 - self.ema_alpha) * self.ema_prob
                            p_used = float(self.ema_prob)
                        else:
                            p_used = float(p_active)

                        vote = 1 if p_used >= self.prob_threshold else 0
                        self.votes.append(vote)
                        votes_sum = sum(self.votes)
                        is_active = (votes_sum >= self.required_votes and len(self.votes) == self.vote_window)
                        label_str = "ACTIVE" if is_active else "REST"

                        # push decision row
                        self.gui_queue.put(("decision", {
                            "p_raw": float(p_active),
                            "p_used": float(p_used),
                            "votes_sum": int(votes_sum),
                            "votes_len": int(len(self.votes)),
                            "label": label_str,
                            "timestamp": time.time()
                        }))
                else:
                    time.sleep(self.idle_sleep)
        except Exception as e:
            self.gui_queue.put(("error", f"Serial worker error: {e}"))
        finally:
            try:
                if self.serial is not None:
                    self.serial.close()
            except Exception:
                pass
            self.gui_queue.put(("info", "Serial closed"))

# -------------------- GUI --------------------

class EMGGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMG Realtime GUI")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.geometry("1100x700")

        self.gui_queue = queue.Queue()
        self.worker = None

        self.raw_history = deque(maxlen=2000)  # raw samples history for plotting
        self.prob_history = deque(maxlen=1000) # p_used history
        self.time_history = deque(maxlen=1000)

        self._build_controls()
        self._build_plots()

        # schedule queue processing
        self.after(50, self._process_queue)

    def _build_controls(self):
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # left: connection & model
        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.X, padx=6)

        ttk.Label(left, text="COM:").grid(row=0, column=0, sticky=tk.W)
        self.com_var = tk.StringVar(value=DEFAULT_COM)
        ttk.Entry(left, textvariable=self.com_var, width=10).grid(row=0, column=1)

        ttk.Label(left, text="Baud:").grid(row=0, column=2, sticky=tk.W)
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
        ttk.Entry(left, textvariable=self.baud_var, width=8).grid(row=0, column=3)

        ttk.Label(left, text="Model:").grid(row=1, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(left, textvariable=self.model_var, width=40).grid(row=1, column=1, columnspan=3, sticky=tk.W)
        ttk.Button(left, text="Browse", command=self._browse_model).grid(row=1, column=4, padx=4)

        ttk.Label(left, text="Scaler:").grid(row=2, column=0, sticky=tk.W)
        self.scaler_var = tk.StringVar(value=DEFAULT_SCALER)
        ttk.Entry(left, textvariable=self.scaler_var, width=40).grid(row=2, column=1, columnspan=3, sticky=tk.W)
        ttk.Button(left, text="Browse", command=self._browse_scaler).grid(row=2, column=4, padx=4)

        ttk.Label(left, text="LabelEnc:").grid(row=3, column=0, sticky=tk.W)
        self.le_var = tk.StringVar(value=DEFAULT_LE)
        ttk.Entry(left, textvariable=self.le_var, width=40).grid(row=3, column=1, columnspan=3, sticky=tk.W)
        ttk.Button(left, text="Browse", command=self._browse_le).grid(row=3, column=4, padx=4)

        # middle: runtime controls
        middle = ttk.Frame(frm)
        middle.pack(side=tk.LEFT, padx=8)

        self.start_btn = ttk.Button(middle, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=4)
        self.stop_btn = ttk.Button(middle, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=4)

        ttk.Label(middle, text="Threshold:").grid(row=1, column=0)
        self.threshold_var = tk.DoubleVar(value=0.85)
        ttk.Scale(middle, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.threshold_var, length=200).grid(row=1, column=1, columnspan=2)
        ttk.Label(middle, textvariable=self.threshold_var).grid(row=1, column=3, padx=6)

        self.ema_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(middle, text="Use EMA", variable=self.ema_var).grid(row=2, column=0)
        ttk.Label(middle, text="EMA alpha:").grid(row=2, column=1)
        self.ema_alpha_var = tk.DoubleVar(value=0.4)
        ttk.Entry(middle, textvariable=self.ema_alpha_var, width=6).grid(row=2, column=2)

        ttk.Label(middle, text="Vote window:").grid(row=3, column=0)
        self.vote_window_var = tk.IntVar(value=4)
        ttk.Entry(middle, textvariable=self.vote_window_var, width=6).grid(row=3, column=1)
        ttk.Label(middle, text="Required votes:").grid(row=3, column=2)
        self.req_votes_var = tk.IntVar(value=2)
        ttk.Entry(middle, textvariable=self.req_votes_var, width=6).grid(row=3, column=3)

        # right: status
        right = ttk.Frame(frm)
        right.pack(side=tk.RIGHT, padx=6)
        ttk.Label(right, text="Status:").pack(anchor=tk.W)
        self.status_text = tk.Text(right, width=40, height=6, state=tk.DISABLED)
        self.status_text.pack()

        # small display for probabilities and label
        display = ttk.Frame(self)
        display.pack(side=tk.TOP, fill=tk.X, padx=8)
        ttk.Label(display, text="p_raw:").pack(side=tk.LEFT)
        self.p_raw_var = tk.StringVar(value="0.000")
        ttk.Label(display, textvariable=self.p_raw_var, width=8).pack(side=tk.LEFT)

        ttk.Label(display, text="p_used:").pack(side=tk.LEFT, padx=(8,0))
        self.p_used_var = tk.StringVar(value="0.000")
        ttk.Label(display, textvariable=self.p_used_var, width=8).pack(side=tk.LEFT)

        ttk.Label(display, text="Votes:").pack(side=tk.LEFT, padx=(8,0))
        self.votes_var = tk.StringVar(value="0/0")
        ttk.Label(display, textvariable=self.votes_var, width=8).pack(side=tk.LEFT)

        ttk.Label(display, text="Label:").pack(side=tk.LEFT, padx=(8,0))
        self.label_var = tk.StringVar(value="-")
        ttk.Label(display, textvariable=self.label_var, width=8, foreground='blue').pack(side=tk.LEFT)

    def _build_plots(self):
        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.fig = Figure(figsize=(9,5))
        self.ax_raw = self.fig.add_subplot(211)
        self.ax_prob = self.fig.add_subplot(212)

        self.ax_raw.set_title('Raw EMG (sliding window)')
        self.ax_raw.set_ylabel('Amplitude')
        self.ax_prob.set_title('Active probability (p_used)')
        self.ax_prob.set_ylabel('Probability')
        self.ax_prob.set_ylim(-0.05, 1.05)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # initial empty lines
        self.raw_line, = self.ax_raw.plot([], [])
        self.prob_line, = self.ax_prob.plot([], [])

        # autoscale flags
        self.raw_xlim = 500

    def _browse_model(self):
        p = filedialog.askopenfilename(title='Select model file', filetypes=[('Joblib','*.joblib;*.pkl;*.sav'),('All','*.*')])
        if p:
            self.model_var.set(p)

    def _browse_scaler(self):
        p = filedialog.askopenfilename(title='Select scaler file', filetypes=[('Pickle','*.pkl;*.joblib'),('All','*.*')])
        if p:
            self.scaler_var.set(p)

    def _browse_le(self):
        p = filedialog.askopenfilename(title='Select label encoder file', filetypes=[('Pickle','*.pkl;*.joblib'),('All','*.*')])
        if p:
            self.le_var.set(p)

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo('Info', 'Worker already running')
            return

        com = self.com_var.get().strip()
        baud = self.baud_var.get().strip()
        model = self.model_var.get().strip() or DEFAULT_MODEL
        scaler = self.scaler_var.get().strip() or None
        le = self.le_var.get().strip() or None

        try:
            prob_threshold = float(self.threshold_var.get())
            use_ema = bool(self.ema_var.get())
            ema_alpha = float(self.ema_alpha_var.get())
            vote_window = int(self.vote_window_var.get())
            req_votes = int(self.req_votes_var.get())
        except Exception as e:
            messagebox.showerror('Error', f'Invalid parameter: {e}')
            return

        self.worker = SerialWorker(com, baud, model, scaler, le, self.gui_queue,
                                   win_size=WIN_SIZE, step_size=STEP_SIZE,
                                   prob_threshold=prob_threshold, use_ema=use_ema,
                                   ema_alpha=ema_alpha, vote_window=vote_window,
                                   required_votes=req_votes)
        self.worker.start()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._log(f"Started worker on {com}@{baud}")

    def stop(self):
        if self.worker is None:
            return
        self.worker.stop()
        self._log('Stopping worker...')
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _process_queue(self):
        processed = False
        while True:
            try:
                item = self.gui_queue.get_nowait()
            except queue.Empty:
                break
            processed = True
            kind = item[0]
            data = item[1]
            if kind == 'sample':
                self.raw_history.append(float(data))
            elif kind == 'decision':
                self.p_raw_var.set(f"{data['p_raw']:.3f}")
                self.p_used_var.set(f"{data['p_used']:.3f}")
                self.votes_var.set(f"{data['votes_sum']}/{data['votes_len']}")
                self.label_var.set(data['label'])
                self.prob_history.append(data['p_used'])
                self.time_history.append(data['timestamp'])
            elif kind == 'info':
                self._log(str(data))
            elif kind == 'error':
                self._log(str(data))
                messagebox.showerror('Error', str(data))
            else:
                self._log(f"Unknown queue message: {item}")

        if processed:
            self._update_plots()

        self.after(50, self._process_queue)

    def _update_plots(self):
        # raw plot: show last N samples
        raw = np.array(self.raw_history)
        if raw.size == 0:
            return
        x_raw = np.arange(raw.size)
        self.raw_line.set_data(x_raw, raw)
        self.ax_raw.set_xlim(max(0, raw.size - self.raw_xlim), raw.size)
        y_min, y_max = raw.min(), raw.max()
        if y_min == y_max:
            y_min -= 0.1
            y_max += 0.1
        self.ax_raw.set_ylim(y_min - 0.01 * abs(y_min), y_max + 0.01 * abs(y_max))

        # prob plot
        prob = np.array(self.prob_history)
        if prob.size > 0:
            x_prob = np.arange(prob.size)
            self.prob_line.set_data(x_prob, prob)
            self.ax_prob.set_xlim(0, max(100, prob.size))

        self.canvas.draw_idle()

    def _log(self, text):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {text}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def on_close(self):
        if messagebox.askokcancel('Quit', 'Stop and quit?'):
            try:
                if self.worker is not None:
                    self.worker.stop()
            except Exception:
                pass
            self.destroy()

# -------------------- entry point --------------------

if __name__ == '__main__':
    app = EMGGUI()
    app.mainloop()
