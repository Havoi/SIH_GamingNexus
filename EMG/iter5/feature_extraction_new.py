import numpy as np
import pandas as pd
from collections import Counter
import glob
import os

# Pattern to match your input streams
INPUT_PATTERN = "labeled_stream*"
OUTPUT_CSV = "overall_features_alldata_resampled.csv"

# ==== EMG SETTINGS (time-based) ====
window_sec = 0.179    # e.g., 179 ms windows
step_sec   = 0.046    # e.g., 46 ms step

# Whether to use a robust target fs selection: 'median' or 'min' or 'max' or a float value
TARGET_FS_SELECTION = "median"  # or set to a number like 200.0

# If you will downsample a lot and want anti-aliasing, set to True (requires scipy)
USE_ANTI_ALIAS_FILTER = False

try:
    if USE_ANTI_ALIAS_FILTER:
        from scipy.signal import butter, filtfilt, decimate
        SCIPY_AVAILABLE = True
    else:
        SCIPY_AVAILABLE = False
except Exception:
    SCIPY_AVAILABLE = False

def estimate_fs_from_timestamps(ts: np.ndarray) -> float:
    dt = np.diff(ts)
    # remove zero or extremely small dt entries if any
    dt = dt[np.isfinite(dt) & (dt > 1e-12)]
    if len(dt) == 0:
        raise ValueError("Insufficient timestamp variation to estimate fs.")
    median_dt = np.median(dt)
    return 1.0 / median_dt

def extract_features_from_window(x: np.ndarray) -> dict:
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

def resample_timeseries(timestamps: np.ndarray, values: np.ndarray, labels: np.ndarray, target_fs: float):
    """
    Resample `values` to a uniform time grid with spacing 1/target_fs between
    first timestamp and last timestamp. Map labels to the new grid by nearest.
    """
    t0 = float(timestamps[0])
    t1 = float(timestamps[-1])
    dt = 1.0 / target_fs
    if t1 <= t0:
        raise ValueError("Timestamps not strictly increasing or identical start/end in file.")
    new_times = np.arange(t0, t1 + 1e-12, dt)

    # linear interpolation for values
    # note: np.interp requires x to be increasing
    new_values = np.interp(new_times, timestamps, values)

    # for labels (categorical), map using nearest original sample
    # we create a pandas Series with original timestamps (index) and labels as values,
    # then reindex with nearest method.
    lab_series = pd.Series(labels, index=pd.Index(timestamps))
    # reindex with nearest; if too far (tolerance) we could fill NaN, but here we use nearest without tolerance
    # convert new_times to same dtype as index
    try:
        new_labels = lab_series.reindex(pd.Index(new_times), method="nearest").values
    except Exception:
        # fallback: for each new time find nearest index via searchsorted (vectorized)
        idx = np.searchsorted(timestamps, new_times)
        idx = np.clip(idx, 1, len(timestamps)-1)
        left = idx - 1
        right = idx
        left_diff = np.abs(new_times - timestamps[left])
        right_diff = np.abs(timestamps[right] - new_times)
        choose_right = right_diff < left_diff
        nearest_idx = left
        nearest_idx[choose_right] = right[choose_right]
        new_labels = labels[nearest_idx]

    return new_times, new_values, new_labels

def process_and_extract(csv_path: str, target_fs: float) -> pd.DataFrame:
    print(f"\nProcessing file: {csv_path}")
    df = pd.read_csv(csv_path)
    # Expect columns: timestamp (float seconds), value (numeric), label (categorical)
    timestamps = df["timestamp"].values.astype(float)
    values = df["value"].values.astype(float)
    labels_raw = df["label"].values

    # Optional anti-aliasing when downsampling: only if scipy available
    # If you want, apply a lowpass filter before resampling if target_fs << file_fs
    # (this script uses simple linear interpolation; for heavy downsampling, use scipy.signal.decimate)
    if SCIPY_AVAILABLE and USE_ANTI_ALIAS_FILTER:
        try:
            fs_file = estimate_fs_from_timestamps(timestamps)
            if fs_file > 2.5 * target_fs:
                # design a lowpass butterworth filter at Nyquist of target_fs
                nyq = 0.5 * fs_file
                cutoff = 0.5 * target_fs  # conservative cutoff
                b, a = butter(4, cutoff / nyq)
                # apply filter to values (filtfilt to avoid phase)
                values = filtfilt(b, a, values)
        except Exception as e:
            print("  Warning: anti-alias filtering skipped:", e)

    new_t, new_v, new_lab = resample_timeseries(timestamps, values, labels_raw, target_fs)
    # compute sample-based window sizes
    win_size = int(round(window_sec * target_fs))
    step_size = int(round(step_sec * target_fs))
    if win_size < 1:
        raise ValueError("window_sec too small for target_fs -> win_size < 1")
    if step_size < 1:
        raise ValueError("step_sec too small for target_fs -> step_size < 1")

    print(f"  Resampled to {len(new_t)} samples at {target_fs:.3f} Hz. win_size={win_size}, step_size={step_size}")

    features = []
    labels = []
    start_times = []
    end_times = []

    n = len(new_t)
    i = 0
    while i + win_size <= n:
        seg = new_v[i:i + win_size]
        seg_labels = new_lab[i:i + win_size]
        seg_times = new_t[i:i + win_size]

        feat = extract_features_from_window(seg)
        if feat is not None:
            features.append(feat)
            # majority label inside the window -- preserves original intention
            maj_label = Counter(seg_labels).most_common(1)[0][0]
            labels.append(maj_label)
            start_times.append(seg_times[0])
            end_times.append(seg_times[-1])
        i += step_size

    feat_df = pd.DataFrame(features)
    feat_df["label"] = labels
    feat_df["start_time"] = start_times
    feat_df["end_time"] = end_times
    feat_df["source_file"] = os.path.basename(csv_path)
    return feat_df

# ==== MAIN: figure target fs first by scanning files ====
input_files = sorted(glob.glob(INPUT_PATTERN))
if not input_files:
    raise FileNotFoundError(f"No files found matching pattern: {INPUT_PATTERN}")

print("Found input files:")
for f in input_files:
    print(" ", f)

# First pass: estimate per-file fs
fs_list = []
for f in input_files:
    df = pd.read_csv(f)
    ts = df["timestamp"].values.astype(float)
    fs = estimate_fs_from_timestamps(ts)
    fs_list.append(fs)
    print(f"  {os.path.basename(f)} -> estimated fs {fs:.2f} Hz")

# Choose target_fs
if isinstance(TARGET_FS_SELECTION, (int, float)):
    target_fs = float(TARGET_FS_SELECTION)
else:
    sel = TARGET_FS_SELECTION.lower()
    if sel == "median":
        target_fs = float(np.median(fs_list))
    elif sel == "min":
        target_fs = float(np.min(fs_list))
    elif sel == "max":
        target_fs = float(np.max(fs_list))
    else:
        raise ValueError("Unknown TARGET_FS_SELECTION: use 'median','min','max' or numeric value")

print(f"\nUsing target_fs = {target_fs:.3f} Hz (selection: {TARGET_FS_SELECTION})")

# Process each file: resample and extract features
all_feature_dfs = []
for f in input_files:
    feat_df = process_and_extract(f, target_fs)
    all_feature_dfs.append(feat_df)

combined_df = pd.concat(all_feature_dfs, ignore_index=True)
combined_df.to_csv(OUTPUT_CSV, index=False)
print("\nSaved combined features to:", OUTPUT_CSV)
print(combined_df.head())
