"""
auto_label_on_mean.py

Auto-label UP when the moving-mean reaches PEAK_HIGH and then falls below it after staying,
and auto-label DOWN when the moving-mean reaches PEAK_LOW and then rises above it after staying.

Writes rows: timestamp, value, event to OUTPUT_CSV and shows live plot with raw + mean lines
and event markers.

Configure parameters in the USER CONFIG section below.
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
COM_PORT = 'COM5'         # change to your serial port
BAUD_RATE = 115200
MAX_SAMPLES = 600         # how many recent samples to display
Y_MIN = 0
Y_MAX = 1024
OUTPUT_CSV = "mean_auto_labeled_stream.csv"

# Mean and detection params
MEAN_WINDOW = 5          # number of most recent samples to compute the mean (odd recommended)
PEAK_HIGH = 1023          # high threshold for mean to be considered "at peak"
PEAK_LOW = 0              # low threshold for mean to be considered "at low peak"
HYSTERESIS = 2            # units to move away before re-arming (prevents immediate re-trigger)
MIN_STAY_MS = 150         # how long (ms) mean must stay at/above peak before we count it as a true touch
DEBOUNCE_MS = 400         # min time between emitted events (ms)
# ===============================================

# Setup CSV
new_file = not os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, "a", newline="")
csv_writer = csv.writer(csv_file)
if new_file:
    csv_writer.writerow(["timestamp", "value", "event"])

# Serial init
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"✅ Connected to {COM_PORT}. Waiting for data...")
    time.sleep(1.0)
except serial.SerialException as e:
    print(f"❌ ERROR: Could not open {COM_PORT}: {e}")
    csv_file.close()
    raise SystemExit(1)

# Buffers for plotting and mean calculation
data_buffer = deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)
time_buffer = deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)

# Thread-safety
lock = threading.Lock()

# State for mean-touch detection
last_event_time_ms = 0
high_in = False             # True while mean is at/above PEAK_HIGH
high_entry_time_ms = None
low_in = False              # True while mean is at/at-or-below PEAK_LOW
low_entry_time_ms = None

# Plot markers (for visual feedback)
event_markers = []  # list of (timestamp, label, sample_index)

def emit_label(ts, val, label, sample_index):
    """Write label row to CSV, print, and add plot marker."""
    global last_event_time_ms
    with lock:
        csv_writer.writerow([f"{ts:.6f}", val, label])
        csv_file.flush()
        # append marker for plotting: sample_index is index from right (0=oldest, -1=latest)
        event_markers.append((ts, label, sample_index))
    last_event_time_ms = int(time.time() * 1000)
    print(f"[{time.strftime('%H:%M:%S')}] AUTO-LABEL -> {label}  value={val}  time={ts:.6f}")

def read_serial_loop():
    global high_in, high_entry_time_ms, low_in, low_entry_time_ms
    while True:
        try:
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                if raw_line == "":
                    continue
                # parse numeric (int or float)
                try:
                    val = float(raw_line) if ('.' in raw_line) else int(raw_line)
                except:
                    continue
                t = time.time()
                with lock:
                    data_buffer.append(float(val))
                    time_buffer.append(t)

                # compute mean over last MEAN_WINDOW samples if available
                with lock:
                    arr = np.array(data_buffer)
                if len(arr) < 1:
                    continue
                # compute mean over last MEAN_WINDOW (or fewer when starting)
                w = min(MEAN_WINDOW, len(arr))
                mean_val = float(np.mean(arr[-w:]))

                now_ms = int(time.time() * 1000)

                # HIGH (UP) detection logic:
                # - when mean >= PEAK_HIGH and not already high_in: mark entry time
                # - when high_in and mean falls below (PEAK_HIGH - HYSTERESIS) and stayed >=PEAK_HIGH for >= MIN_STAY_MS -> emit UP
                if not high_in and mean_val >= PEAK_HIGH:
                    high_in = True
                    high_entry_time_ms = now_ms
                elif high_in:
                    # still high?
                    if mean_val < (PEAK_HIGH - HYSTERESIS):
                        # left the peak region
                        duration_ms = now_ms - (high_entry_time_ms or now_ms)
                        if duration_ms >= MIN_STAY_MS and (now_ms - last_event_time_ms) > DEBOUNCE_MS:
                            # emit event at drop moment; capture latest (raw) value as representative
                            with lock:
                                sample_idx = len(data_buffer) - 1
                                sample_val = data_buffer[-1]
                                sample_ts = time_buffer[-1] if len(time_buffer) else t
                            emit_label(sample_ts, sample_val, "UP", sample_idx)
                        # reset
                        high_in = False
                        high_entry_time_ms = None

                # LOW (DOWN) detection logic:
                # - when mean <= PEAK_LOW and not low_in: mark entry time
                # - when low_in and mean rises above (PEAK_LOW + HYSTERESIS) and stayed <=PEAK_LOW for >= MIN_STAY_MS -> emit DOWN
                if not low_in and mean_val <= PEAK_LOW:
                    low_in = True
                    low_entry_time_ms = now_ms
                elif low_in:
                    if mean_val > (PEAK_LOW + HYSTERESIS):
                        duration_ms = now_ms - (low_entry_time_ms or now_ms)
                        if duration_ms >= MIN_STAY_MS and (now_ms - last_event_time_ms) > DEBOUNCE_MS:
                            with lock:
                                sample_idx = len(data_buffer) - 1
                                sample_val = data_buffer[-1]
                                sample_ts = time_buffer[-1] if len(time_buffer) else t
                            emit_label(sample_ts, sample_val, "DOWN", sample_idx)
                        low_in = False
                        low_entry_time_ms = None

        except Exception as e:
            # on unexpected errors, don't crash; simple delay and continue reading
            print("Serial read error:", e)
            time.sleep(0.01)

# Plotting setup
fig, ax = plt.subplots(figsize=(10, 4))
raw_line, = ax.plot([0]*MAX_SAMPLES, lw=0.8, label='raw')
mean_line, = ax.plot([0]*MAX_SAMPLES, lw=2.2, label=f'mean({MEAN_WINDOW})')
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_title(f"Mean-based auto-labeling ({COM_PORT}) — UP when mean touches {PEAK_HIGH} then drops; DOWN when mean touches {PEAK_LOW} then rises")
ax.set_ylabel("Signal")
ax.set_xlabel("Samples (recent on right)")
ax.grid(True)
ax.legend(loc='upper right')

# vertical markers drawn dynamically
marker_lines = []

def update_plot(frame):
    with lock:
        y = list(data_buffer)
        tbuf = list(time_buffer)
        markers = list(event_markers)  # copy
    if not y:
        return raw_line, mean_line

    # compute mean trace (moving mean over MEAN_WINDOW)
    arr = np.array(y)
    n = len(arr)
    w = min(MEAN_WINDOW, n)
    mean_trace = np.convolve(arr, np.ones(w)/w, mode='valid')
    # pad left to same length as arr
    pad = n - len(mean_trace)
    mean_full = np.concatenate([np.full(pad, mean_trace[0]), mean_trace]) if len(mean_trace)>0 else np.zeros(n)

    x = np.arange(len(y))
    raw_line.set_data(x, y)
    mean_line.set_data(x, mean_full)
    # cleanup old marker lines
    for ml in marker_lines:
        try:
            ml.remove()
        except:
            pass
    marker_lines.clear()
    # draw markers: for each event, draw vertical line at its sample index (if still in view)
    # event_markers entries contain sample_index at time of emission (0..len-1)
    with lock:
        current_len = len(data_buffer)
        for (evt_ts, label, sample_idx) in markers:
            # compute x position: sample_idx (index into buffer at time of emission)
            # If sample_idx might be older than buffer length, skip (we only have relative positions)
            # We'll attempt to map index to current window by offsetting relative to end:
            # sample_idx here was len(buffer)-1 at emission; we stored absolute sample_idx at that time.
            # To approximate, compute offset from end at emission: offset = (current_len - 1) - sample_idx_at_emission
            # But we stored sample_idx as absolute index at emission, and buffer is rolling, so better approach:
            # Instead, we stored sample_idx as snapshot index at emission; we will draw marker at the right-most edge (latest) for clarity.
            # Simpler: draw marker at latest sample (x = n-1)
            x_pos = len(y) - 1
            color = 'green' if label == 'UP' else 'red'
            ml, = ax.plot([x_pos, x_pos], [Y_MIN, Y_MAX], lw=1.0, linestyle='--', color=color, alpha=0.6)
            marker_lines.append(ml)

    ax.relim()
    ax.autoscale_view()
    return raw_line, mean_line

# Start serial reader thread
reader_thread = threading.Thread(target=read_serial_loop, daemon=True)
reader_thread.start()

# Start animation
ani = animation.FuncAnimation(fig, update_plot, interval=40, blit=True)
print("📈 Plotting... Close the graph window to stop. Mean-based auto-labeling is active.")
try:
    plt.show()
finally:
    try:
        ser.close()
    except:
        pass
    csv_file.close()
    print("Stopped. CSV saved to", OUTPUT_CSV)