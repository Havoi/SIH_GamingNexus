import serial
import time
from collections import deque
import numpy as np
import joblib

# ================== USER CONFIG =====================
COM_PORT = "COM5"            # change if needed
BAUD_RATE = 115200

MODEL_PATH = "emg_wrist_rf_model.pkl"
LABEL_ENCODER_PATH = "emg_label_encoder.pkl"

# Window settings (MUST match how you trained features)
# Example: if you used ~5 Hz sampling and 1s window => win_size ~ 5
WIN_SIZE = 446    # number of samples per window
STEP_SIZE = 22     # slide window by this many samples
# ====================================================


# === EMG feature extraction (same as training) ===
def extract_emg_features(x: np.ndarray) -> np.ndarray:
    """
    x: 1D numpy array of EMG samples in a window.
    Returns: feature vector [MAV, RMS, VAR, WL, ZC, SSC, WAMP, IEMG]
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0:
        return None

    # center and rectify
    x_centered = x - np.mean(x)
    x_abs = np.abs(x_centered)

    std = np.std(x_centered)
    thr = 0.05 * std if std > 0 else 0.0

    mav = np.mean(x_abs)
    rms = np.sqrt(np.mean(x_centered ** 2))
    var = np.var(x_centered)
    wl = np.sum(np.abs(np.diff(x_centered)))

    # zero crossings
    zc = np.sum(
        ((x_centered[:-1] * x_centered[1:]) < 0) &
        (np.abs(np.diff(x_centered)) > thr)
    )

    # slope sign changes
    dx1 = np.diff(x_centered)
    ssc = np.sum(
        ((dx1[:-1] * dx1[1:]) < 0) &
        (np.abs(dx1[:-1]) > thr) &
        (np.abs(dx1[1:]) > thr)
    )

    # Willison amplitude
    wamp = np.sum(np.abs(np.diff(x_centered)) > thr)

    # Integrated EMG
    iemg = np.sum(x_abs)

    return np.array([mav, rms, var, wl, zc, ssc, wamp, iemg], dtype=float)


def main():
    # Load model + label encoder
    print("Loading model and label encoder...")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("Classes:", le.classes_)

    # Open serial port
    print(f"Opening serial port {COM_PORT} @ {BAUD_RATE}...")
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2.0)  # allow Arduino reset

    # Buffer for recent samples
    buffer = deque(maxlen=WIN_SIZE)
    sample_count = 0
    last_label = None

    print("✅ Ready. Reading EMG and predicting in real time...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # Read one line from serial
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                # Try parse int/float
                try:
                    if "." in raw_line:
                        val = float(raw_line)
                    else:
                        val = int(raw_line)
                except ValueError:
                    # ignore non-numeric lines
                    continue

                buffer.append(float(val))
                sample_count += 1

                # Only predict when we have enough samples and after each STEP_SIZE steps
                if len(buffer) == WIN_SIZE and (sample_count % STEP_SIZE == 0):
                    window = np.array(buffer, dtype=float)
                    feat_vec = extract_emg_features(window)
                    if feat_vec is None:
                        continue

                    # Model expects shape (1, n_features)
                    feat_vec_2d = feat_vec.reshape(1, -1)
                    y_pred_int = model.predict(feat_vec_2d)[0]
                    label = le.inverse_transform([y_pred_int])[0]

                    # Only print when label changes (less spam)
                    if label != last_label:
                        t = time.strftime("%H:%M:%S")
                        print(f"[{t}] Prediction: {label}")
                        last_label = label

            else:
                # No data, small sleep to avoid busy loop
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()