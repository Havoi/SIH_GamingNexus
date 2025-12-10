#!/usr/bin/env python3
"""
EMG Realtime GUI — Dark theme, uses features.py and model_wrapper.py

Place this file in the same folder as:
 - features.py        (provides extract_emg_features)
 - model_wrapper.py   (provides ModelWrapper)

Behavior changes from your original:
 - EMG model workflow expects an MLP model + scaler + label-encoder (three files).
 - Sliding vote window: we append one vote for each decision and let deque(maxlen=window) slide it.
 - Threshold slider updates the threshold line AND recomputes votes immediately.
 - Dark theme applied to ttk and matplotlib.

Run:
    python emg_realtime_dark.py

Dependencies:
    pip install pyserial joblib numpy matplotlib
"""
from __future__ import annotations
import threading
import queue
import time
from collections import deque
import numpy as np
import os
import sys
import keyboard
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# local modular helpers (must exist)
from features import extract_emg_features
from model_wrapper import ModelWrapper

# Serial (only used when not simulating)
import serial

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


# ---------- Background worker (serial or simulated) ----------
class SerialWorker(threading.Thread):
    """Background thread that reads serial (or simulated), computes features & probabilities and pushes updates to GUI queue."""

    def __init__(self, com, baud, model_path, scaler_path, le_path, gui_queue,
                 win_size=WIN_SIZE, step_size=STEP_SIZE,
                 use_ema=True, ema_alpha=0.4,
                 vote_window=4, required_votes=2, idle_sleep=0.001,
                 simulate=False, sim_rate_hz=100):
        super().__init__(daemon=True)
        self.com = com
        self.baud = int(baud)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.le_path = le_path
        self.gui_queue = gui_queue
        self.win_size = int(win_size)
        self.step_size = int(step_size)
        self.use_ema = bool(use_ema)
        self.ema_alpha = float(ema_alpha)
        self.vote_window = int(vote_window)
        self.required_votes = int(required_votes)
        self.idle_sleep = float(idle_sleep)

        self.simulate = bool(simulate)
        self.sim_rate_hz = float(sim_rate_hz)

        self._stop_event = threading.Event()

        # runtime
        self.buffer = deque(maxlen=self.win_size)
        self.sample_count = 0
        self.votes = deque([], maxlen=self.vote_window)
        self.ema_prob = None

        self.serial = None
        self.model_wrapper: ModelWrapper | None = None

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def _open_serial(self):
        try:
            self.serial = serial.Serial(self.com, self.baud, timeout=1)
            return True, None
        except Exception as e:
            return False, e

    def _load_model(self):
        # We load using ModelWrapper which accepts model+scaler+le
        try:
            # require all three for EMG MLP workflow
            if not (self.model_path and self.scaler_path and self.le_path):
                raise RuntimeError("EMG model, scaler and label-encoder must be provided")
            self.model_wrapper = ModelWrapper(self.model_path, scaler_path=self.scaler_path, le_path=self.le_path)
            return True, None
        except Exception as e:
            return False, e

    def run(self):
        ok, err = self._load_model()
        if not ok:
            self.gui_queue.put(("error", f"Model load failed: {err}"))
            return

        if self.simulate:
            # simulation loop: emit a float at sim_rate_hz
            dt = 1.0 / max(1.0, self.sim_rate_hz)
            t = 0.0
            burst_period = 3.0
            while not self.stopped():
                # produce synthetic EMG: small noise + occasional bursts
                burst = 1.0 if (int(t) % int(burst_period) == 0) else 0.0
                val = 0.02 * np.random.randn() + burst * (0.5 * np.random.rand())
                self.buffer.append(float(val))
                self.sample_count += 1
                self.gui_queue.put(("sample", float(val)))

                if len(self.buffer) == self.win_size and (self.sample_count % self.step_size == 0):
                    feats = extract_emg_features(np.array(self.buffer))
                    feats_2d = feats.reshape(1, -1)
                    try:
                        p_active = float(self.model_wrapper.predict_proba_active(feats_2d))
                    except Exception:
                        p_active = 0.0

                    # EMA
                    if self.use_ema:
                        if self.ema_prob is None:
                            self.ema_prob = p_active
                        else:
                            self.ema_prob = self.ema_alpha * p_active + (1.0 - self.ema_alpha) * self.ema_prob
                        p_used = float(self.ema_prob)
                    else:
                        p_used = float(p_active)

                    # produce decision and push to GUI
                    self.gui_queue.put(("decision", {
                        "p_raw": float(p_active),
                        "p_used": float(p_used),
                        "timestamp": time.time()
                    }))

                time.sleep(dt)
                t += dt

            return

        # real serial mode
        ok, err = self._open_serial()
        if not ok:
            self.gui_queue.put(("error", f"Serial open failed: {err}"))
            return
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
                    self.gui_queue.put(("sample", float(val)))

                    if len(self.buffer) == self.win_size and (self.sample_count % self.step_size == 0):
                        feats = extract_emg_features(np.array(self.buffer))
                        feats_2d = feats.reshape(1, -1)
                        try:
                            p_active = float(self.model_wrapper.predict_proba_active(feats_2d))
                        except Exception:
                            print("probabilty set to 0")
                            p_active = 0.0

                        # EMA
                        if self.use_ema:
                            if self.ema_prob is None:
                                self.ema_prob = p_active
                            else:
                                self.ema_prob = self.ema_alpha * p_active + (1.0 - self.ema_alpha) * self.ema_prob
                            p_used = float(self.ema_prob)
                        else:
                            p_used = float(p_active)

                        self.gui_queue.put(("decision", {
                            "p_raw": float(p_active),
                            "p_used": float(p_used),
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

# dark theme helper
import keyboard
import time

    
    
def apply_dark_ttk_style(root):
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass

    bg = '#0f1114'
    fg = '#e6eef6'
    entry_bg = '#0b0d10'
    btn_bg = '#172027'

    style.configure('.', background=bg, foreground=fg)
    style.configure('TLabel', background=bg, foreground=fg)
    style.configure('TButton', background=btn_bg, foreground=fg)
    style.configure('TEntry', fieldbackground=entry_bg, foreground=fg)
    style.configure('TFrame', background=bg)
    root.configure(bg=bg)


class EMGGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMG Realtime (Dark)")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.geometry("1100x680")

        apply_dark_ttk_style(self)

        self.gui_queue = queue.Queue()
        self.worker = None

        # histories
        self.raw_history = deque(maxlen=2000)
        self.prob_history = deque(maxlen=1000)
        self.time_history = deque(maxlen=1000)

        # votes deque (sliding)
        self.vote_window = 4
        self.req_votes = 2
        self.votes = deque([], maxlen=self.vote_window)

        # EMA state
        self.ema_prob = None

        self._build_controls()
        self._build_plots()

        # trace threshold changes for realtime effect
        self.threshold_var.trace_add('write', self._on_threshold_change)
        # trace vote window edits
        self.vote_window_var.trace_add('write', self._on_vote_window_change)

        self.after(40, self._process_queue)

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

        ttk.Label(left, text="Model (MLP):").grid(row=1, column=0, sticky=tk.W)
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
        self.vote_window_var = tk.IntVar(value=self.vote_window)
        ttk.Entry(middle, textvariable=self.vote_window_var, width=6).grid(row=3, column=1)
        ttk.Label(middle, text="Required votes:").grid(row=3, column=2)
        self.req_votes_var = tk.IntVar(value=self.req_votes)
        ttk.Entry(middle, textvariable=self.req_votes_var, width=6).grid(row=3, column=3)

        # right: status
        right = ttk.Frame(frm)
        right.pack(side=tk.RIGHT, padx=6)
        ttk.Label(right, text="Status:").pack(anchor=tk.W)
        self.status_text = tk.Text(right, width=40, height=6, state=tk.DISABLED, bg='#0b0d10', fg='#e6eef6')
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
        ttk.Label(display, textvariable=self.label_var, width=8, foreground='cyan').pack(side=tk.LEFT)

    def _build_plots(self):
        plot_frame = ttk.Frame(self)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.fig = Figure(figsize=(9,5), facecolor='#0b0d10')
        self.ax_raw = self.fig.add_subplot(211, facecolor='#0a0c0f')
        self.ax_prob = self.fig.add_subplot(212, facecolor='#0a0c0f')

        # styling
        for ax in (self.ax_raw, self.ax_prob):
            ax.title.set_color('#e6eef6')
            ax.xaxis.label.set_color('#e6eef6')
            ax.yaxis.label.set_color('#e6eef6')
            ax.tick_params(colors='#bfcbdc')
            for sp in ax.spines.values():
                sp.set_color('#26292d')

        self.ax_raw.set_title('Raw EMG (sliding window)')
        self.ax_raw.set_ylabel('Amplitude')
        self.ax_prob.set_title('Active probability (p_used)')
        self.ax_prob.set_ylabel('Probability')
        self.ax_prob.set_ylim(-0.05, 1.05)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # initial empty lines (high contrast)
        self.raw_line, = self.ax_raw.plot([], [], color='#7ee787')
        self.prob_line, = self.ax_prob.plot([], [], color='#4FD1C5')

        # threshold line (store and update when slider moves)
        self.threshold_line = self.ax_prob.axhline(self.threshold_var.get(), linestyle='--', color='#ffb86b')

        # autoscale flags
        self.raw_xlim = 500

    def _browse_model(self):
        p = filedialog.askopenfilename(title='Select model file (MLP joblib)', filetypes=[('Joblib','*.joblib;*.pkl;*.sav'),('All','*.*')])
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
        # verify model/scaler/le present
        model = self.model_var.get().strip() or DEFAULT_MODEL
        scaler = self.scaler_var.get().strip() or DEFAULT_SCALER
        le = self.le_var.get().strip() or DEFAULT_LE
        com = self.com_var.get().strip()
        baud = self.baud_var.get().strip()

        try:
            # parameters (only used inside worker for EMA settings)
            use_ema = bool(self.ema_var.get())
            ema_alpha = float(self.ema_alpha_var.get())
            vote_window = int(self.vote_window_var.get())
            req_votes = int(self.req_votes_var.get())
        except Exception as e:
            messagebox.showerror('Error', f'Invalid parameter: {e}')
            return

        # if worker running, ignore
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo('Info', 'Worker already running')
            return

        # create worker (simulate if COM blank)
        simulate = (not com) or (com.lower() == 'simulate') or (com.strip() == '')
        self.worker = SerialWorker(com, baud, model, scaler, le, self.gui_queue,
                                   win_size=WIN_SIZE, step_size=STEP_SIZE,
                                   use_ema=use_ema, ema_alpha=ema_alpha,
                                   vote_window=vote_window, required_votes=req_votes,
                                   simulate=simulate, sim_rate_hz=100)
        self.worker.start()

        # initialize votes deque with configured window
        self.vote_window = vote_window
        self.req_votes = req_votes
        self.votes = deque([], maxlen=self.vote_window)

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._log(f"Started worker on {com or 'SIMULATE'}@{baud or 'sim'}")

    def stop(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker = None
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
                # incoming decision contains p_raw and p_used and timestamp
                p_raw = float(data.get('p_raw', 0.0))
                p_used = float(data.get('p_used', 0.0))
                ts = data.get('timestamp', time.time())

                # update histories
                self.prob_history.append(p_used)
                self.time_history.append(ts)

                # append a single vote based on current threshold -> sliding window
                try:
                    thr = float(self.threshold_var.get())
                except Exception:
                    thr = 0.85
                new_vote = 1 if p_used >= thr else 0

                # ensure vote deque length matches UI-configured window
                try:
                    desired_win = int(self.vote_window_var.get())
                except Exception:
                    desired_win = self.votes.maxlen or 4
                if self.votes.maxlen != desired_win:
                    old = list(self.votes)
                    self.votes = deque(old[-desired_win:], maxlen=desired_win)

                self.votes.append(new_vote)
                votes_sum = sum(self.votes)
                votes_len = len(self.votes)

                # decision: do not require deque to be full
                is_active = votes_sum >= int(self.req_votes_var.get())
                label_str = "ACTIVE" if is_active else "REST"

                # update small displays
                self.p_raw_var.set(f"{p_raw:.3f}")
                self.p_used_var.set(f"{p_used:.3f}")
                self.votes_var.set(f"{votes_sum}/{votes_len}")
                self.label_var.set(label_str)

            elif kind == 'info':
                self._log(str(data))
            elif kind == 'error':
                self._log(str(data))
                messagebox.showerror('Error', str(data))
            else:
                self._log(f"Unknown queue message: {item}")

        if processed:
            self._update_plots()

        self.after(40, self._process_queue)

    def _update_plots(self):
        # raw plot: show last N samples
        raw = np.array(self.raw_history)
        if raw.size > 0:
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

        # threshold line update is done elsewhere (trace) but ensure ydata is current
        try:
            thr = float(self.threshold_var.get())
            self.threshold_line.set_ydata([thr, thr])
        except Exception:
            pass

        self.canvas.draw_idle()

    # realtime threshold callback: recompute votes from recent probabilities immediately
    def _on_threshold_change(self, *_):
        try:
            thr = float(self.threshold_var.get())
        except Exception:
            return

        # recompute votes from the most recent probabilities
        try:
            vote_window = int(self.vote_window_var.get())
        except Exception:
            vote_window = self.votes.maxlen or 4
        recent_probs = list(self.prob_history)[-vote_window:]
        recent_votes = [1 if p >= thr else 0 for p in recent_probs]
        self.votes = deque(recent_votes, maxlen=vote_window)

        votes_sum = sum(self.votes)
        votes_len = len(self.votes)
        self.votes_var.set(f"{votes_sum}/{votes_len}")
        try:
            last_p = float(self.p_used_var.get())
        except Exception:
            last_p = recent_probs[-1] if recent_probs else 0.0
        self.p_used_var.set(f"{last_p:.3f}")
        # update threshold line visually
        try:
            self.threshold_line.set_ydata([thr, thr])
        except Exception:
            pass
        self.canvas.draw_idle()

    def _on_vote_window_change(self, *_):
        # preserve recent votes while resizing window
        try:
            new_win = int(self.vote_window_var.get())
            if new_win <= 0:
                return
        except Exception:
            return
        old = list(self.votes)
        self.votes = deque(old[-new_win:], maxlen=new_win)
        # recompute counts with current threshold
        self._on_threshold_change()

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
