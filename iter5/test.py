import time
from collections import deque
import numpy as np
import joblib
import serial

COM_PORT = "COM5"
BAUD_RATE = 115200
MODEL_PATH = "emg_rest_active_rf_model.pkl"
LABEL_ENCODER_PATH = "emg_rest_active_label_encoder.pkl"
WIN_SIZE = 89
STEP_SIZE = 22

def extract_emg_features(x: np.ndarray):
    x = np.asarray(x, dtype=float)
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
        zc = np.sum(((x_centered[:-1] * x_centered[1:]) < 0) & (np.abs(np.diff(x_centered)) > thr))
    ssc = 0
    if x_centered.size > 2:
        dx1 = np.diff(x_centered)
        ssc = np.sum(((dx1[:-1] * dx1[1:]) < 0) & (np.abs(dx1[:-1]) > thr) & (np.abs(dx1[1:]) > thr))
    wamp = np.sum(np.abs(np.diff(x_centered)) > thr) if x_centered.size > 1 else 0
    iemg = np.sum(x_abs)
    return np.array([mav, rms, var, wl, zc, ssc, wamp, iemg], dtype=float), {'min': x.min(), 'max': x.max(), 'mean': x.mean(), 'std': std, 'thr': thr}

model = joblib.load(MODEL_PATH)
try:
    le = joblib.load(LABEL_ENCODER_PATH)
except:
    le = None

ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
buffer = deque(maxlen=WIN_SIZE)
sample_count = 0
print("Starting diagnostics — perform rest and then an active contraction while watching output")
try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            try:
                v = float(line)
            except:
                continue
            buffer.append(v)
            sample_count += 1
            if len(buffer) == WIN_SIZE and (sample_count % STEP_SIZE == 0):
                win = np.array(buffer)
                feats, stats = extract_emg_features(win)
                feats2 = feats.reshape(1, -1)
                pred = model.predict(feats2)[0]
                probs = None
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(feats2)[0]
                    except:
                        probs = None
                if le is not None:
                    try:
                        label = le.inverse_transform([pred])[0]
                    except:
                        label = str(pred)
                else:
                    label = str(pred)
                probstr = ""
                if probs is not None:
                    probstr = " PROBS:" + ", ".join([f"{p:.2f}" for p in probs])
                print(f"{time.strftime('%H:%M:%S')} RAWmin={stats['min']:.2f} RAWmax={stats['max']:.2f} mean={stats['mean']:.2f} std={stats['std']:.4f} thr={stats['thr']:.4f} MAV={feats[0]:.4f} RMS={feats[1]:.4f} VAR={feats[2]:.4f} -> {str(label).upper()}{probstr}")
        else:
            time.sleep(0.001)
except KeyboardInterrupt:
    ser.close()
    print("Stopped")