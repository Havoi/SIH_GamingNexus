#!/usr/bin/env python3
"""
EMG realtime predictor (terminal output only)
- Loads model, scaler, label encoder
- Extracts features from sliding windows
- Uses predict_proba (when available), EMA smoothing and voting for accurate ACTIVE detection
- Prints timestamp, probability and decided label to terminal
"""

import time
from collections import deque
import numpy as np
import joblib
import serial
import sys
import os

# ============== USER CONFIG ================
COM_PORT = "COM5"
BAUD_RATE = 115200

MODEL_PATH = "emg_mlp_model_clean.joblib"
SCALER_PATH = "emg_scaler_clean.pkl"            # optional
LABEL_ENCODER_PATH = "emg_label_encoder_clean.pkl"  # optional

WIN_SIZE = 89
STEP_SIZE = 22

ACTIVE_LABEL = "ACTIVE"   # case-insensitive target label

# Accuracy-first tuning (increase to reduce false positives)
PROB_THRESHOLD = 0.85
USE_EMA = True
EMA_ALPHA = 0.4

VOTE_WINDOW = 4
REQUIRED_VOTES = 2

# small sleep when serial idle
IDLE_SLEEP = 0.001
# ============================================


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
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print("Failed to load model:", e)
        raise

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


def get_active_probability(model, feats, le):
    """
    Return float probability (0..1) that feats correspond to ACTIVE.
    Tries predict_proba first; falls back to predict-based heuristics.
    """
    # try predict_proba
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(feats)  # (1, C)
            classes = None
            # prefer label encoder classes if provided
            if le is not None and hasattr(le, "classes_"):
                classes = le.classes_
            elif hasattr(model, "classes_"):
                classes = model.classes_

            if classes is not None:
                # find index of ACTIVE (case-insensitive)
                idx = None
                for i, c in enumerate(classes):
                    try:
                        if str(c).strip().upper() == ACTIVE_LABEL:
                            idx = i
                            break
                    except Exception:
                        continue
                if idx is not None:
                    return float(probs[0, idx])
                # fallback: if binary, take class 1
                if probs.shape[1] == 2:
                    return float(probs[0, 1])
            # last fallback: return max prob
            return float(np.max(probs[0]))
        except Exception:
            pass

    # fallback: model.predict
    try:
        pred = model.predict(feats)
        p = pred[0] if isinstance(pred, (list, tuple, np.ndarray)) else pred
        if le is not None:
            try:
                lbl = str(le.inverse_transform([p])[0]).strip().upper()
                return 1.0 if lbl == ACTIVE_LABEL else 0.0
            except Exception:
                pass
        if isinstance(p, str):
            return 1.0 if p.strip().upper() == ACTIVE_LABEL else 0.0
        if hasattr(model, "classes_"):
            classes = model.classes_
            try:
                if isinstance(p, (int, np.integer)) and 0 <= int(p) < len(classes):
                    lbl = str(classes[int(p)]).strip().upper()
                    return 1.0 if lbl == ACTIVE_LABEL else 0.0
            except Exception:
                pass
    except Exception:
        pass

    return 0.0


def main():
    print("Loading model/scaler/encoder...")
    try:
        model, scaler, le = load_model_scaler_encoder(MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH)
    except Exception:
        sys.exit(1)

    # print classes if possible
    try:
        if le is not None and hasattr(le, "classes_"):
            print("Label encoder classes:", le.classes_)
        elif hasattr(model, "classes_"):
            print("Model classes:", model.classes_)
    except Exception:
        pass

    # open serial
    print(f"Opening serial {COM_PORT} @ {BAUD_RATE} ...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print("Serial open failed:", e)
        sys.exit(1)

    time.sleep(2.0)  # allow device to settle

    buffer = deque(maxlen=WIN_SIZE)
    sample_count = 0

    votes = deque(maxlen=VOTE_WINDOW)
    ema_prob = None
    last_printed_state = None

    print("\nReady — printing predictions to terminal. Ctrl+C to stop.\n")

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    val = float(line)
                except Exception:
                    continue

                buffer.append(val)
                sample_count += 1

                if len(buffer) == WIN_SIZE and (sample_count % STEP_SIZE == 0):
                    feats = extract_emg_features(np.array(buffer))
                    if feats is None:
                        continue
                    feats_2d = feats.reshape(1, -1)

                    # scale if scaler present
                    if scaler is not None:
                        try:
                            feats_in = scaler.transform(feats_2d)
                        except Exception:
                            feats_in = feats_2d
                    else:
                        feats_in = feats_2d

                    p_active = get_active_probability(model, feats_in, le)

                    # EMA smoothing
                    if USE_EMA:
                        if ema_prob is None:
                            ema_prob = p_active
                        else:
                            ema_prob = EMA_ALPHA * p_active + (1.0 - EMA_ALPHA) * ema_prob
                        p_used = float(ema_prob)
                    else:
                        p_used = float(p_active)

                    # vote and decide
                    vote = 1 if p_used >= PROB_THRESHOLD else 0
                    votes.append(vote)
                    votes_sum = sum(votes)
                    is_active = (votes_sum >= REQUIRED_VOTES and len(votes) == VOTE_WINDOW)

                    # prepare printed label and info
                    label_str = "ACTIVE" if is_active else "REST"
                    t = time.strftime("%H:%M:%S")

                    # Print every window (or optionally only on change) — here we print every decision row
                    print(f"[{t}] p_raw={p_active:.3f} p_used={p_used:.3f} votes={votes_sum}/{len(votes)} => {label_str}")

                    last_printed_state = label_str

            else:
                time.sleep(IDLE_SLEEP)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("Serial closed.")


if __name__ == "__main__":
    main()
