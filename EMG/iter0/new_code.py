"""
label_wrist_smoothed.py

Improved live plot + light-weight denoising of Arduino wrist signal.

Features added:
- real-time moving-average + EMA smoothing
- optional running-median baseline removal (helps if baseline drifts)
- shows raw trace (thin) and smoothed trace (thick)
- keeps CSV labeling on Up/Down (same as you had)
- toggle smoothing on/off with 't' key
- adjustable parameters at the top

Requires: numpy, matplotlib, pyserial
pip install numpy matplotlib pyserial
"""

import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import csv
import os
import numpy as np

# ================= USER CONFIG =================
COM_PORT = 'COM5'
BAUD_RATE = 115200
MAX_SAMPLES = 500        # history shown on screen
Y_MIN = 0
Y_MAX = 1024
OUTPUT_CSV = "labeled_stream.csv"
DEBOUNCE_SECONDS = 0.2

# Denoising/tuning params (experiment with these)
MA_WINDOW = 7            # moving-average window (odd recommended, >=1). Larger -> smoother, more lag
EMA_ALPHA = 0.15         # EMA smoothing factor (0..1). Lower -> smoother/slower
BASELINE_MEDIAN_WINDOW = 201  # odd integer in samples; set to 1 to disable baseline removal
# =================================================

# CSV setup
new_file = not os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, "a", newline="")
csv_writer = csv.writer(csv_file)
if new_file:
    csv_writer.writerow(["timestamp", "value", "event", "smoothed_value"])

# Serial init
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"✅ Connected to {COM_PORT}. Waiting for data...")
    time.sleep(2)
except serial.SerialException:
    print(f"❌ ERROR: Could not open {COM_PORT}.")
    print("👉 Close Serial Monitor / other programs using the port.")
    csv_file.close()
    exit()

# Buffers
data_buffer = deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)
time_buffer = deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)
smoothed_buffer = deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)

# State
lock = threading.Lock()
last_label_time = 0.0
smoothing_enabled = True
ema_state = None

# Helper: simple moving average using numpy (works on small arrays quickly)
def moving_average(arr, window):
    if window <= 1:
        return arr.copy()
    # 1D convolution
    kernel = np.ones(window) / window
    # pad to keep same length
    padded = np.pad(arr, (window//2, window-1-window//2), mode='edge')
    sm = np.convolve(padded, kernel, mode='valid')
    return sm

# Helper: running median baseline using numpy (simple, not super optimized but fine for small buffers)
def running_median(arr, window):
    if window <= 1:
        return np.zeros_like(arr)
    n = len(arr)
    med = np.zeros(n)
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        med[i] = np.median(arr[start:end])
    return med

# Serial read thread
def read_serial():
    global ema_state
    while True:
        try:
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                if raw_line == "":
                    continue
                # accept ints or floats
                try:
                    if '.' in raw_line:
                        val = float(raw_line)
                    else:
                        val = int(raw_line)
                except:
                    continue
                t = time.time()
                with lock:
                    data_buffer.append(float(val))
                    time_buffer.append(t)
                    # update EMA incrementally for the latest sample only (fast, minimal computation)
                    if ema_state is None:
                        ema_state = float(val)
                    else:
                        ema_state = EMA_ALPHA * float(val) + (1.0 - EMA_ALPHA) * ema_state
                    smoothed_buffer.append(float(ema_state))
        except Exception as e:
            # don't spam on errors; continue reading
            # print("Serial read error:", e)
            time.sleep(0.005)

# Plot setup
fig, ax = plt.subplots()
raw_line, = ax.plot([], [], lw=0.8, label='raw', alpha=0.9)
smooth_line, = ax.plot([], [], lw=2.2, label='smoothed', alpha=0.9)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_title(f"Real-Time EMG Signal ({COM_PORT})  — Press Up/Down or 'u'/'d' to label; 't' toggles smoothing")
ax.set_ylabel("Signal Amplitude")
ax.set_xlabel("Samples (most recent on right)")
ax.grid(True)
ax.legend(loc='upper right')

# Update function: compute smoothed trace from buffer and show both traces
def update_plot(frame):
    with lock:
        y_raw = np.array(data_buffer)
        tbuf = list(time_buffer)
    if y_raw.size == 0:
        return raw_line, smooth_line

    if smoothing_enabled:
        # 1) optionally remove baseline using running median (slower but robust)
        if BASELINE_MEDIAN_WINDOW > 1 and len(y_raw) >= BASELINE_MEDIAN_WINDOW:
            baseline = running_median(y_raw, BASELINE_MEDIAN_WINDOW)
            centered = y_raw - baseline
        else:
            centered = y_raw - np.median(y_raw)  # small DC remove

        # 2) moving average (FIR) to remove high-frequency jitter
        ma = moving_average(centered, MA_WINDOW)

        # 3) EMA on top of moving average (further smoothing with low lag)
        # We'll apply EMA in vectorized form: y_ema[i] = alpha*ma[i] + (1-alpha)*y_ema[i-1]
        y_ema = np.zeros_like(ma)
        alpha = EMA_ALPHA
        y_ema[0] = ma[0]
        for i in range(1, len(ma)):
            y_ema[i] = alpha * ma[i] + (1.0 - alpha) * y_ema[i-1]

        # put back median to shift to similar scale as raw (optional)
        y_smooth = y_ema + np.median(y_raw)
    else:
        # smoothing disabled: show EMA incremental buffer (fast)
        with lock:
            y_smooth = np.array(smoothed_buffer)

    # update plot data (ensure same lengths)
    x = np.arange(len(y_raw))
    # If lengths mismatch due to small initial buffers, align them
    n = min(len(x), len(y_smooth))
    raw_line.set_data(x[-n:], y_raw[-n:])
    smooth_line.set_data(x[-n:], y_smooth[-n:])
    ax.relim()
    ax.autoscale_view()
    return raw_line, smooth_line

# Key handler for labeling + toggling smoothing
def on_key(event):
    global last_label_time, smoothing_enabled
    key = event.key
    label = None
    if key in ('up', 'u'):
        label = 'UP'
    elif key in ('down', 'd'):
        label = 'DOWN'
    elif key == 't':
        smoothing_enabled = not smoothing_enabled
        print("Smoothing toggled ->", smoothing_enabled)
        return
    else:
        return

    now = time.time()
    if now - last_label_time < DEBOUNCE_SECONDS:
        print("Debounced: press ignored.")
        return

    with lock:
        if len(time_buffer) == 0 or len(data_buffer) == 0:
            print("No data to label yet.")
            return
        ts = float(time_buffer[-1])
        raw_val = float(data_buffer[-1])
        sm_val = float(smoothed_buffer[-1]) if len(smoothed_buffer) > 0 else ''
    csv_writer.writerow([f"{ts:.6f}", raw_val, label, f"{sm_val}"])
    csv_file.flush()
    last_label_time = now
    print(f"Label recorded: {label} at {ts:.6f} raw={raw_val} smoothed={sm_val}")

# Wire up
fig.canvas.mpl_connect('key_press_event', on_key)
thread = threading.Thread(target=read_serial, daemon=True)
thread.start()

# Start animation
ani = animation.FuncAnimation(fig, update_plot, interval=30, blit=True)
print("📈 Plotting... Close the graph window to stop. Use Up/Down (or u/d) to label, 't' to toggle smoothing.")
try:
    plt.show()
finally:
    try:
        ser.close()
    except:
        pass
    csv_file.close()
    print("Disconnected and CSV saved.")