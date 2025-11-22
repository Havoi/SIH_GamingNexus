"""
label_wrist_smoothed_3class.py
Real-time EMG plot + continuous labeling: REST / UP / DOWN

Controls:
- 'r' → REST
- 'u' or Up arrow → UP
- 'd' or Down arrow → DOWN
- 't' → toggle extra smoothing

Each incoming sample is written to CSV with the current label.
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
OUTPUT_CSV = "labeled_stream_3class.csv"

# Denoising/tuning params
MA_WINDOW = 7                 # moving-average window
EMA_ALPHA = 0.15              # EMA smoothing factor
BASELINE_MEDIAN_WINDOW = 21  # odd integer; 1 to disable
# =================================================

# CSV setup
new_file = not os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, "a", newline="")
csv_writer = csv.writer(csv_file)
if new_file:
    csv_writer.writerow(["timestamp", "value", "label", "smoothed_value"])

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
smoothing_enabled = True
ema_state = None

# NEW: current label (REST by default)
current_label = "REST"   # will be changed with keys r/u/d


# Helper: simple moving average
def moving_average(arr, window):
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window//2, window-1-window//2), mode='edge')
    sm = np.convolve(padded, kernel, mode='valid')
    return sm


# Helper: running median baseline
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
                try:
                    if '.' in raw_line:
                        val = float(raw_line)
                    else:
                        val = int(raw_line)
                except:
                    continue

                t = time.time()

                with lock:
                    # update buffers
                    data_buffer.append(float(val))
                    time_buffer.append(t)

                    # incremental EMA (fast)
                    if ema_state is None:
                        ema_state = float(val)
                    else:
                        ema_state = EMA_ALPHA * float(val) + (1.0 - EMA_ALPHA) * ema_state
                    smoothed_buffer.append(float(ema_state))

                    # get current label snapshot
                    label_snapshot = current_label
                    sm_val = float(ema_state)

                # Write to CSV (outside lock to keep lock section short)
                csv_writer.writerow([f"{t:.6f}", float(val), label_snapshot, f"{sm_val}"])
                # You can comment this out if file I/O is too heavy
                csv_file.flush()

        except Exception as e:
            # print("Serial read error:", e)
            time.sleep(0.005)


# Plot setup
fig, ax = plt.subplots()
raw_line, = ax.plot([], [], lw=0.8, label='raw', alpha=0.9)
smooth_line, = ax.plot([], [], lw=2.2, label='smoothed', alpha=0.9)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_title(
    f"Real-Time EMG Signal ({COM_PORT})  — "
    "r=REST, u/↑=UP, d/↓=DOWN, t=toggle smoothing"
)
ax.set_ylabel("Signal Amplitude")
ax.set_xlabel("Samples (most recent on right)")
ax.grid(True)
ax.legend(loc='upper right')


# Update function: compute smoothed trace from buffer and show both traces
def update_plot(frame):
    with lock:
        y_raw = np.array(data_buffer)
    if y_raw.size == 0:
        return raw_line, smooth_line

    if smoothing_enabled:
        if BASELINE_MEDIAN_WINDOW > 1 and len(y_raw) >= BASELINE_MEDIAN_WINDOW:
            baseline = running_median(y_raw, BASELINE_MEDIAN_WINDOW)
            centered = y_raw - baseline
        else:
            centered = y_raw - np.median(y_raw)

        ma = moving_average(centered, MA_WINDOW)

        y_ema = np.zeros_like(ma)
        alpha = EMA_ALPHA
        y_ema[0] = ma[0]
        for i in range(1, len(ma)):
            y_ema[i] = alpha * ma[i] + (1.0 - alpha) * y_ema[i-1]

        y_smooth = y_ema + np.median(y_raw)
    else:
        with lock:
            y_smooth = np.array(smoothed_buffer)

    x = np.arange(len(y_raw))
    n = min(len(x), len(y_smooth))
    raw_line.set_data(x[-n:], y_raw[-n:])
    smooth_line.set_data(x[-n:], y_smooth[-n:])
    ax.relim()
    ax.autoscale_view()
    return raw_line, smooth_line


# Key handler for changing label + toggling smoothing
def on_key(event):
    global smoothing_enabled, current_label
    key = event.key

    if key == 't':
        smoothing_enabled = not smoothing_enabled
        print("Smoothing toggled ->", smoothing_enabled)
        return

    # Label keys
    if key in ('r',):
        current_label = "REST"
    elif key in ('up', 'u'):
        current_label = "UP"
    elif key in ('down', 'd'):
        current_label = "DOWN"
    else:
        return

    print(f"🔖 Current label set to: {current_label}")


# Wire up
fig.canvas.mpl_connect('key_press_event', on_key)
thread = threading.Thread(target=read_serial, daemon=True)
thread.start()

ani = animation.FuncAnimation(fig, update_plot, interval=30, blit=True)
print("📈 Plotting... Close the graph window to stop.")
print("Controls: r=REST, u/↑=UP, d/↓=DOWN, t=toggle smoothing")

try:
    plt.show()
finally:
    try:
        ser.close()
    except:
        pass
    csv_file.close()
    print("Disconnected and CSV saved.")