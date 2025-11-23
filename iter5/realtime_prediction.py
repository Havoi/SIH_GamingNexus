#!/usr/bin/env python3
"""
EMG -> Chrome Dino bot (simplified prediction)
Predicts ACTIVE vs REST using a single .pkl model file (joblib).
Triggers Space when model predicts ACTIVE.
"""

import time
from collections import deque
import numpy as np
import joblib
import serial
import pyautogui
import sys
import os

# ================== USER CONFIG (edit if needed) =====================
COM_PORT = "COM5"            # serial port to read from
BAUD_RATE = 115200

# Model files (model should be a joblib pickle)
MODEL_PATH = "emg_rest_active_rf_model.pkl"
LABEL_ENCODER_PATH = "emg_rest_active_label_encoder.pkl"  # optional

# Window settings (must match training)
WIN_SIZE = 89    # number of samples per window
STEP_SIZE = 22   # slide window by this many samples

# Label that counts as "active" (comparison is case-insensitive)
ACTIVE_LABELS = {"ACTIVE"}

# Refractory and jump settings
REFRACTORY_PERIOD_SEC = 1.5
JUMP_HOLD_SEC = 0.06
# ===================================================================

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
    """Send a short Space keypress."""
    try:
        prev_pause = pyautogui.PAUSE
        pyautogui.PAUSE = 0
        pyautogui.keyDown('space')
        time.sleep(JUMP_HOLD_SEC)
        pyautogui.keyUp('space')
        pyautogui.PAUSE = prev_pause
    except Exception as e:
        print("Error sending jump key:", e)


def load_model_and_encoder(model_path: str, encoder_path: str = None):
    """Load ML model and optional label encoder. Return (model, le_or_None)."""
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model from '{model_path}':", e)
        raise

    le = None
    if encoder_path and os.path.exists(encoder_path):
        try:
            le = joblib.load(encoder_path)
        except Exception as e:
            print(f"Warning: failed to load label encoder '{encoder_path}':", e)
            le = None

    return model, le


def decode_label(pred, model, le):
    """
    Decode prediction to a string label.
    pred: raw model.predict(...) output (single element or scalar)
    model: loaded model (might have classes_ attribute)
    le: optional label encoder (inverse_transform)
    """
    # If prediction is a 1-element array or list, extract
    if isinstance(pred, (list, tuple, np.ndarray)):
        if len(pred) == 0:
            return None
        p = pred[0]
    else:
        p = pred

    # If label encoder provided, try to inverse transform numeric or encoded label
    if le is not None:
        try:
            return str(le.inverse_transform([p])[0])
        except Exception:
            # fallthrough to other strategies
            pass

    # If model.predict already returns string labels, just return them
    if isinstance(p, str):
        return p

    # If model has classes_ attribute and p is integer index, map it
    if hasattr(model, "classes_"):
        try:
            # if p is index within classes_ range
            if isinstance(p, (int, np.integer)) and 0 <= int(p) < len(model.classes_):
                return str(model.classes_[int(p)])
            # Sometimes model.predict yields class labels directly (already handled), so try to match value
            # Try to find exact match among classes_
            for c in model.classes_:
                if np.array_equal(c, p) or str(c) == str(p):
                    return str(c)
        except Exception:
            pass

    # As a last resort, stringify the prediction
    return str(p)


def main():
    print("Loading model and optional label encoder...")
    try:
        model, le = load_model_and_encoder(MODEL_PATH, LABEL_ENCODER_PATH)
    except Exception:
        sys.exit(1)

    # Show available classes if possible
    try:
        if le is not None and hasattr(le, "classes_"):
            print("Label encoder classes:", le.classes_)
        elif hasattr(model, "classes_"):
            print("Model classes:", model.classes_)
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

    time.sleep(2.0)  # allow device reset

    buffer = deque(maxlen=WIN_SIZE)
    sample_count = 0
    last_label_printed = None
    last_active_time = None

    print("\n✅ Ready. Focus Chrome (Dino) window. Press Ctrl+C to stop.\n")

    try:
        while True:
            # Read serial if available
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

                    # Predict
                    try:
                        y_pred = model.predict(feat_vec_2d)
                    except Exception as e:
                        print("Model predict error:", e)
                        continue

                    label = decode_label(y_pred, model, le)
                    if label is None:
                        continue

                    label_str = str(label).strip().upper()
                    now = time.time()

                    if label_str in ACTIVE_LABELS:
                        if last_active_time is None or (now - last_active_time) >= REFRACTORY_PERIOD_SEC:
                            t_hms = time.strftime("%H:%M:%S")
                            print(f"[{t_hms}] ACTIVE -> JUMP ({label})")
                            last_active_time = now
                            last_label_printed = label_str
                            send_jump()
                        else:
                            # suppressed due to refractory
                            pass
                    else:
                        # REST or other — print when label changes
                        if label_str != last_label_printed:
                            t_hms = time.strftime("%H:%M:%S")
                            print(f"[{t_hms}] {label}")
                            last_label_printed = label_str
            else:
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
