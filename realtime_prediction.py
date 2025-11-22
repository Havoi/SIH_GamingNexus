#!/usr/bin/env python3
"""
EMG -> Chrome Dino bot
Presses Space when your EMG model predicts an "active" label (UP/DOWN).
Based on the EMG feature extraction and main loop you provided.
"""

import time
from collections import deque
import numpy as np
import joblib
import serial
import pyautogui
import sys
import platform

# ================== USER CONFIG (edit if needed) =====================
COM_PORT = "COM5"            # serial port to read from
BAUD_RATE = 115200

MODEL_PATH = "emg_wrist_rf_model.pkl"
LABEL_ENCODER_PATH = "emg_label_encoder.pkl"

# Window settings (MUST match how you trained features)
WIN_SIZE = 446    # number of samples per window
STEP_SIZE = 22    # slide window by this many samples

# Active labels that should be treated/displayed as "active"
ACTIVE_LABELS = {"UP", "DOWN"}  # case-insensitive comparison

# Refractory period: once an active (UP/DOWN) is printed & jumped, suppress further active prints for this many seconds
REFRACTORY_PERIOD_SEC = 1.5

# How long to hold the jump key (seconds) — short hold is enough for Dino jump
JUMP_HOLD_SEC = 0.06
# =====================================================================

# === EMG feature extraction (same as training) ===
def extract_emg_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
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


def send_jump():
    """
    Send a Space key press (short hold). Works when the Chrome window/tab with Dino is focused.
    """
    try:
        # On some platforms pyautogui has a small pause by default. We'll temporarily disable failsafe pause,
        # press, then restore.
        prev_pause = pyautogui.PAUSE
        pyautogui.PAUSE = 0
        # press and hold:
        pyautogui.keyDown('space')
        time.sleep(JUMP_HOLD_SEC)
        pyautogui.keyUp('space')
        pyautogui.PAUSE = prev_pause
    except Exception as e:
        print("Error sending jump key:", e)


def main():
    # Load model + encoder
    print("Loading model and label encoder...")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)
        sys.exit(1)
    try:
        le = joblib.load(LABEL_ENCODER_PATH)
    except Exception:
        le = None

    try:
        if le is not None:
            print("Classes:", getattr(le, "classes_", "unavailable"))
    except Exception:
        pass

    # Open serial port
    print(f"Opening serial port {COM_PORT} @ {BAUD_RATE}...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print("Failed to open serial port:", e)
        print("Make sure COM_PORT is correct and device is connected.")
        sys.exit(1)

    # small delay for Arduino reset
    time.sleep(2.0)

    # Buffers and state
    buffer = deque(maxlen=WIN_SIZE)
    sample_count = 0
    last_label_printed = None
    last_active_time = None  # timestamp of last printed UP/DOWN event

    print("\n✅ Ready. Put the Chrome window with the Dino in front (focused).")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                try:
                    val = float(raw_line)
                except Exception:
                    # ignore non-numeric lines
                    continue

                buffer.append(val)
                sample_count += 1

                if len(buffer) == WIN_SIZE and (sample_count % STEP_SIZE == 0):
                    window = np.array(buffer, dtype=float)
                    feat_vec = extract_emg_features(window)
                    if feat_vec is None:
                        continue

                    feat_vec_2d = feat_vec.reshape(1, -1)
                    try:
                        y_pred = model.predict(feat_vec_2d)[0]
                    except Exception as e:
                        print("Model predict error:", e)
                        continue

                    # Convert predicted encoding to text label using label encoder if possible
                    try:
                        label = le.inverse_transform([y_pred])[0] if le is not None else str(y_pred)
                    except Exception:
                        label = str(y_pred)

                    label_str = str(label).upper()  # normalize for comparison
                    now = time.time()

                    if label_str in ACTIVE_LABELS:
                        if last_active_time is None or (now - last_active_time) >= REFRACTORY_PERIOD_SEC:
                            t_hms = time.strftime("%H:%M:%S")
                            print(f"[{t_hms}] ACTIVE -> JUMP: {label_str}")
                            last_active_time = now
                            last_label_printed = label_str

                            # Trigger jump
                            send_jump()
                        else:
                            # suppressed due to refractory
                            pass
                    else:
                        # Non-active — print when label changes
                        if label_str != last_label_printed:
                            t_hms = time.strftime("%H:%M:%S")
                            print(f"[{t_hms}] {label_str}")
                            last_label_printed = label_str
            else:
                # small sleep when no data
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("Serial port closed.")


if __name__ == "__main__":
    main()