import numpy as np
import pandas as pd
from collections import Counter
import glob
import os

# Pattern to match all your input streams
# e.g., labeled_stream_1.csv, labeled_stream_2.csv, labeled_stream_2class.csv, etc.
INPUT_PATTERN = "labeled_stream_2class_anirudh_randomnoise.csv"

# Output combined features file
OUTPUT_CSV = "emg_features_combined_2class_anirudh_randomnoise.csv"

# ==== EMG SETTINGS (time-based, independent of exact fs) ====
window_sec = 0.198    # 200 ms window
step_sec   = 0.05     # 50 ms step

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

def extract_features_from_file(csv_path: str) -> pd.DataFrame:
    print(f"\nProcessing file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Estimate sampling rate for this file
    ts = df["timestamp"].values
    dt = np.diff(ts)
    median_dt = np.median(dt)
    fs = 1.0 / median_dt
    print("  Estimated sampling rate:", fs, "Hz")

    win_size  = int(window_sec * fs)
    step_size = int(step_sec * fs)

    print("  win_size  =", win_size,  "samples")
    print("  step_size =", step_size, "samples")

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

    # Add a column to remember which stream/file this came from
    feat_df["source_file"] = os.path.basename(csv_path)

    return feat_df

# ==== MAIN: loop over all labeled_stream*.csv and combine ====
all_feature_dfs = []

input_files = sorted(glob.glob(INPUT_PATTERN))
if not input_files:
    raise FileNotFoundError(f"No files found matching pattern: {INPUT_PATTERN}")

print("Found input files:")
for f in input_files:
    print(" ", f)

for f in input_files:
    feat_df = extract_features_from_file(f)
    all_feature_dfs.append(feat_df)

combined_df = pd.concat(all_feature_dfs, ignore_index=True)

combined_df.to_csv(OUTPUT_CSV, index=False)
print("\nSaved combined features to:", OUTPUT_CSV)
print(combined_df.head())
