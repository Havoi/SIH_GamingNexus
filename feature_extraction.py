import numpy as np
import pandas as pd
from collections import Counter

INPUT_CSV = "labeled_stream_3class.csv"
OUTPUT_CSV = "emg_features_3class.csv"

# 1. Load data
df = pd.read_csv(INPUT_CSV)

# 2. Infer sampling rate from timestamps (assumed in seconds)
ts = df["timestamp"].values
dt = np.diff(ts)
median_dt = np.median(dt)
fs = 1.0 / median_dt
print("Estimated sampling rate:", fs, "Hz")

# 3. Define window + step (in seconds)
window_sec = 4      # 1-second window
step_sec = 0.3        # 50% overlap

win_size = int(window_sec * fs)
step_size = int(step_sec * fs)

def extract_features_from_window(x: np.ndarray) -> dict:
    """Compute standard EMG time-domain features for one window."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0:
        return None

    # center and rectify
    x_centered = x - np.mean(x)
    x_abs = np.abs(x_centered)

    # adaptive threshold for ZC / SSC / WAMP
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

    return {
        "MAV": mav,
        "RMS": rms,
        "VAR": var,
        "WL": wl,
        "ZC": zc,
        "SSC": ssc,
        "WAMP": wamp,
        "IEMG": iemg,
    }

values = df["value"].values
labels_raw = df["label"].values
timestamps = df["timestamp"].values

features = []
labels = []
start_times = []
end_times = []

n = len(df)
i = 0
while i + win_size <= n:
    seg = values[i:i + win_size]
    seg_labels = labels_raw[i:i + win_size]
    seg_times = timestamps[i:i + win_size]

    feat = extract_features_from_window(seg)
    if feat is not None:
        features.append(feat)

        # majority label inside the window
        cnt = Counter(seg_labels)
        maj_label = cnt.most_common(1)[0][0]
        labels.append(maj_label)

        start_times.append(seg_times[0])
        end_times.append(seg_times[-1])

    i += step_size

feat_df = pd.DataFrame(features)
feat_df["label"] = labels
feat_df["start_time"] = start_times
feat_df["end_time"] = end_times

feat_df.to_csv(OUTPUT_CSV, index=False)
print("Saved features to:", OUTPUT_CSV)
print(feat_df.head())