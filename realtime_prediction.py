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
WIN_SIZE = 446    # number of samples per window
STEP_SIZE = 22    # slide window by this many samples

# Active labels that should be treated/displayed as "active"
ACTIVE_LABELS = {"UP", "DOWN"}  # case-insensitive comparison

# Refractory period: once an active (UP/DOWN) is printed, suppress further active prints for this many seconds
REFRACTORY_PERIOD_SEC = 1
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


def main():
    # Load model + encoder
    print("Loading model and label encoder...")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    try:
        print("Classes:", le.classes_)
    except Exception:
        print("Label encoder loaded (classes unavailable).")

    # Open serial port
    print(f"Opening serial port {COM_PORT} @ {BAUD_RATE}...")
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2.0)  # allow Arduino reset

    # Buffers and state
    buffer = deque(maxlen=WIN_SIZE)
    sample_count = 0
    last_label_printed = None

    # For refractory handling of active events
    last_active_time = None  # timestamp of last printed UP/DOWN event

    print("✅ Ready. Reading EMG and predicting in real time...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # Read one line from serial
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                # Try parse numeric value
                try:
                    val = float(raw_line)
                except Exception:
                    # ignore non-numeric lines
                    continue

                buffer.append(val)
                sample_count += 1

                # Only predict when we have enough samples and after each STEP_SIZE steps
                if len(buffer) == WIN_SIZE and (sample_count % STEP_SIZE == 0):
                    window = np.array(buffer, dtype=float)
                    feat_vec = extract_emg_features(window)
                    if feat_vec is None:
                        continue

                    # Model expects shape (1, n_features)
                    feat_vec_2d = feat_vec.reshape(1, -1)
                    y_pred = model.predict(feat_vec_2d)[0]

                    # Convert predicted encoding to text label using label encoder if possible
                    try:
                        label = le.inverse_transform([y_pred])[0]
                    except Exception:
                        label = str(y_pred)

                    label_str = str(label).upper()  # normalize for comparison

                    now = time.time()

                    # If predicted label is one of active labels (UP/DOWN)
                    if label_str in ACTIVE_LABELS:
                        # If within refractory period, suppress active prints
                        if last_active_time is None or (now - last_active_time) >= REFRACTORY_PERIOD_SEC:
                            # Print active event (UP or DOWN)
                            t_hms = time.strftime("%H:%M:%S")
                            # Use original-case label for display if you prefer; here we print the normalized form
                            print(f"[{t_hms}] ACTIVE: {label_str}")
                            last_active_time = now
                            last_label_printed = label_str
                        else:
                            # suppressed due to refractory; do not update last_label_printed
                            pass
                    else:
                        # Non-active (e.g., REST or other) — print when label changes
                        if label_str != last_label_printed:
                            t_hms = time.strftime("%H:%M:%S")
                            print(f"[{t_hms}] {label_str}")
                            last_label_printed = label_str

            else:
                # No data, small sleep to avoid busy loop
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("Serial port closed.")


if __name__== "__main__":
    main()