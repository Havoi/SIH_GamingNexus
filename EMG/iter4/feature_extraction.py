import numpy as np
import pandas as pd
from collections import Counter

INPUT_CSV = "labeled_stream_2class.csv"
OUTPUT_CSV = "emg_features_2class.csv"

# Load data
df = pd.read_csv(INPUT_CSV)

# Estimate sampling rate
ts = df["timestamp"].values
dt = np.diff(ts)
median_dt = np.median(dt)
fs = 1.0 / median_dt
print("Estimated sampling rate:", fs, "Hz")

# ==== EMG SETTINGS FOR 500 Hz ====
window_sec = 0.20     # 200 ms window
step_sec   = 0.05     # 50 ms step

win_size  = int(window_sec * fs)   # ~100 samples
step_size = int(step_sec * fs)     # ~25 samples

print("win_size =", win_size, "samples")
print("step_size =", step_size, "samples")

def extract_features_from_window(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
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
        maj_label = Counter(seg_labels).most_common(1)[0][0]
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