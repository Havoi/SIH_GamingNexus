"""
auto_speak_on_mean_robust.py

Mean-based "speak up/down" with robust TTS worker (pyttsx3 + platform fallback).
Drop-in replacement for the previous script.

Requires: pip install pyttsx3 pyserial matplotlib numpy
"""

import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import os
import numpy as np
import queue
import pyttsx3
import platform
import subprocess
import sys
import traceback

# ================= USER CONFIG =================
COM_PORT = 'COM5'         # change to your serial port
BAUD_RATE = 115200
MAX_SAMPLES = 600         # how many recent samples to display
Y_MIN = 0
Y_MAX = 1024

# Mean and detection params (same as before)
MEAN_WINDOW = 21
PEAK_HIGH = 1023
PEAK_LOW = 0
HYSTERESIS = 2
MIN_STAY_MS = 150
DEBOUNCE_MS = 400
# ===============================================

# Serial init
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"✅ Connected to {COM_PORT}. Waiting for data...")
    time.sleep(1.0)
except serial.SerialException as e:
    print(f"❌ ERROR: Could not open {COM_PORT}: {e}")
    raise SystemExit(1)

# Buffers
data_buffer = deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)
time_buffer = deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)

# Thread-safety & state
lock = threading.Lock()
last_event_time_ms = 0
high_in = False
high_entry_time_ms = None
low_in = False
low_entry_time_ms = None
event_markers = []  # (ts,label)

# TTS queue & worker (robust)
tts_queue = queue.Queue()
_stop_tts = object()  # sentinel

def platform_fallback_speak(text):
    """Best-effort platform TTS as fallback (non-blocking)."""
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.Popen(['say', text])
            return True
        elif system == 'Windows':
            # Use PowerShell SAPI (non-blocking)
            # PowerShell command: Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('text')
            cmd = ["powershell", "-Command",
                   "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{}')".format(text.replace("'", "''"))]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        else:
            # try espeak on Linux (if installed)
            subprocess.Popen(['espeak', text])
            return True
    except Exception as e:
        print("Platform fallback TTS failed:", e)
        return False

def tts_worker():
    """
    Robust TTS worker:
    - creates a pyttsx3 engine and speaks items from queue
    - if the engine errors, re-initialize a few times, and after repeated failures fall back to platform-specific subprocess TTS
    - never dies permanently on a single error
    """
    engine = None
    fail_count = 0
    while True:
        try:
            item = tts_queue.get()
            if item is _stop_tts:
                break
            text = item
            # Lazy init engine if none
            if engine is None:
                try:
                    engine = pyttsx3.init()
                    # tune a bit
                    try:
                        engine.setProperty('rate', 150)
                    except Exception:
                        pass
                    fail_count = 0
                except Exception as e:
                    print("pyttsx3 init failed:", e)
                    engine = None
                    fail_count += 1

            if engine is not None:
                try:
                    # speak the text (blocking inside worker, non-blocking to main program)
                    print(f"[TTS] speaking: {text}")
                    engine.say(text)
                    engine.runAndWait()
                    # small pause to avoid overwhelming TTS engine with rapid queued items
                    time.sleep(0.02)
                    fail_count = 0
                except Exception as e:
                    # engine failed mid-run; dispose and attempt fallback/reinit
                    print("pyttsx3 runtime error:", e)
                    traceback.print_exc()
                    try:
                        engine.stop()
                    except:
                        pass
                    engine = None
                    fail_count += 1
                    # attempt immediate platform fallback for this text
                    ok = platform_fallback_speak(text)
                    if not ok:
                        print("[TTS] fallback also failed for:", text)
            else:
                # engine couldn't be created; use fallback
                ok = platform_fallback_speak(text)
                if not ok:
                    print("[TTS] fallback failed; dropping text:", text)

            # If we've had many consecutive failures, sleep a bit to avoid busy-looping
            if fail_count >= 3:
                print("[TTS] multiple failures detected, pausing TTS worker briefly and will retry engine init")
                time.sleep(1.0)
                fail_count = 0
        except Exception as e:
            print("Unexpected error in tts_worker:", e)
            traceback.print_exc()
            time.sleep(0.2)

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak_async(text):
    """Enqueue text for speaking (non-blocking)."""
    try:
        tts_queue.put_nowait(text)
        print("[SPEAK QUEUED]", text)
    except queue.Full:
        print("TTS queue full; dropped:", text)

def emit_event(ts, val, label):
    global last_event_time_ms
    with lock:
        event_markers.append((ts, label))
    last_event_time_ms = int(time.time() * 1000)
    print(f"[{time.strftime('%H:%M:%S')}] EVENT -> {label}  value={val}  time={ts:.6f}")
    # enqueue speech
    speak_async(label.lower())

def read_serial_loop():
    global high_in, high_entry_time_ms, low_in, low_entry_time_ms
    while True:
        try:
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                if raw_line == "":
                    continue
                try:
                    val = float(raw_line) if ('.' in raw_line) else int(raw_line)
                except:
                    continue
                t = time.time()
                with lock:
                    data_buffer.append(float(val))
                    time_buffer.append(t)

                # compute mean
                with lock:
                    arr = np.array(data_buffer)
                if len(arr) < 1:
                    continue
                w = min(MEAN_WINDOW, len(arr))
                mean_val = float(np.mean(arr[-w:]))

                now_ms = int(time.time() * 1000)

                # HIGH detection
                if not high_in and mean_val >= PEAK_HIGH:
                    high_in = True
                    high_entry_time_ms = now_ms
                elif high_in:
                    if mean_val < (PEAK_HIGH - HYSTERESIS):
                        duration_ms = now_ms - (high_entry_time_ms or now_ms)
                        if duration_ms >= MIN_STAY_MS and (now_ms - last_event_time_ms) > DEBOUNCE_MS:
                            with lock:
                                sample_val = data_buffer[-1] if len(data_buffer) else val
                                sample_ts = time_buffer[-1] if len(time_buffer) else t
                            emit_event(sample_ts, sample_val, "UP")
                        high_in = False
                        high_entry_time_ms = None

                # LOW detection
                if not low_in and mean_val <= PEAK_LOW:
                    low_in = True
                    low_entry_time_ms = now_ms
                elif low_in:
                    if mean_val > (PEAK_LOW + HYSTERESIS):
                        duration_ms = now_ms - (low_entry_time_ms or now_ms)
                        if duration_ms >= MIN_STAY_MS and (now_ms - last_event_time_ms) > DEBOUNCE_MS:
                            with lock:
                                sample_val = data_buffer[-1] if len(data_buffer) else val
                                sample_ts = time_buffer[-1] if len(time_buffer) else t
                            emit_event(sample_ts, sample_val, "DOWN")
                        low_in = False
                        low_entry_time_ms = None
        except Exception as e:
            print("Serial read error:", e)
            traceback.print_exc()
            time.sleep(0.01)

# Plotting (same as before)
fig, ax = plt.subplots(figsize=(10, 4))
raw_line, = ax.plot([0]*MAX_SAMPLES, lw=0.8, label='raw')
mean_line, = ax.plot([0]*MAX_SAMPLES, lw=2.2, label=f'mean({MEAN_WINDOW})')
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_title(f"Mean-based speak-on-event ({COM_PORT}) — speaks 'up' or 'down'")
ax.set_ylabel("Signal")
ax.set_xlabel("Samples (recent on right)")
ax.grid(True)
ax.legend(loc='upper right')
marker_lines = []

def update_plot(frame):
    with lock:
        y = list(data_buffer)
        markers = list(event_markers)
    if not y:
        return raw_line, mean_line
    arr = np.array(y)
    n = len(arr)
    w = min(MEAN_WINDOW, n)
    mean_trace = np.convolve(arr, np.ones(w)/w, mode='valid')
    pad = n - len(mean_trace)
    mean_full = np.concatenate([np.full(pad, mean_trace[0]), mean_trace]) if len(mean_trace) > 0 else np.zeros(n)
    x = np.arange(len(y))
    raw_line.set_data(x, y)
    mean_line.set_data(x, mean_full)
    # clear old markers
    for ml in marker_lines:
        try: ml.remove()
        except: pass
    marker_lines.clear()
    # draw recent event markers at right edge for feedback
    for (evt_ts, label) in markers[-8:]:
        x_pos = len(y) - 1
        color = 'green' if label == 'UP' else 'red'
        ml, = ax.plot([x_pos, x_pos], [Y_MIN, Y_MAX], lw=1.0, linestyle='--', color=color, alpha=0.6)
        marker_lines.append(ml)
    ax.relim()
    ax.autoscale_view()
    return raw_line, mean_line

# start threads
reader_thread = threading.Thread(target=read_serial_loop, daemon=True)
reader_thread.start()
ani = animation.FuncAnimation(fig, update_plot, interval=40, blit=True)

print("📈 Plotting... Close the graph window to stop. The computer will speak 'up'/'down' on events.")
try:
    plt.show()
finally:
    # cleanup
    try:
        ser.close()
    except:
        pass
    # stop TTS thread
    tts_queue.put(_stop_tts)
    # give it a second to exit gracefully
    tts_thread.join(timeout=1.0)
    print("Stopped.")