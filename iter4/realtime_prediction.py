import serial
import time
from collections import deque
import numpy as np
import joblib

# ================== USER CONFIG =====================
COM_PORT = "COM5"            # change if needed
BAUD_RATE = 115200

MODEL_PATH = "emg_rest_active_rf_model.pkl"
LABEL_ENCODER_PATH = "emg_rest_active_label_encoder.pkl"

# Window settings (MUST match training)
WIN_SIZE = 100    # number of samples per window
STEP_SIZE = 25    # slide window by this many samples
# ====================================================


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

    zc = np.sum(
        ((x_centered[:-1] * x_centered[1:]) < 0) &
        (np.abs(np.diff(x_centered)) > thr)
    )

    dx1 = np.diff(x_centered)
    ssc = np.sum(
        ((dx1[:-1] * dx1[1:]) < 0) &
        (np.abs(dx1[:-1]) > thr) &
        (np.abs(dx1[1:]) > thr)
    )

    wamp = np.sum(np.abs(np.diff(x_centered)) > thr)
    iemg = np.sum(x_abs)

    return np.array([mav, rms, var, wl, zc, ssc, wamp, iemg], dtype=float)


def main():
    # Load model + encoder
    print("Loading model and label encoder...")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("Classes:", le.classes_)

    print(f"Opening serial port {COM_PORT} @ {BAUD_RATE}...")
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)

    buffer = deque(maxlen=WIN_SIZE)
    sample_count = 0
    last_label = None

    print("Ready. Reading EMG and predicting...\n")

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # parse numeric
                try:
                    val = float(line)
                except:
                    continue

                buffer.append(val)
                sample_count += 1

                # Predict every STEP_SIZE samples
                if len(buffer) == WIN_SIZE and (sample_count % STEP_SIZE == 0):
                    feat = extract_emg_features(np.array(buffer))
                    if feat is None:
                        continue

                    pred_int = model.predict(feat.reshape(1, -1))[0]
                    label = le.inverse_transform([pred_int])[0]

                    # print only when label changes
                    if label != last_label:
                        t = time.strftime("%H:%M:%S")
                        print(f"[{t}] Prediction: {label}")
                        last_label = label

            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()