"""
features.py

Feature extraction helpers for EMG and IMU.
Keep lightweight, deterministic, and unit-test friendly.

Functions:
- extract_emg_features(x): returns 8 time-domain EMG features
- extract_imu_features(acc, gyro): returns 12 IMU features (6 for acc, 6 for gyro)

"""
from __future__ import annotations
import numpy as np


def extract_emg_features(x: np.ndarray) -> np.ndarray:
    """Extract 8 time-domain EMG features from a 1-D sample window.

    Features returned (in order):
      [mav, rms, var, wl, zc, ssc, wamp, iemg]

    - x: 1-D array-like of samples (float)
    - returns: 1-D numpy array of length 8 (dtype=float)
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(8, dtype=float)

    x_centered = x - np.mean(x)
    x_abs = np.abs(x_centered)
    std = np.std(x_centered)
    thr = 0.05 * std if std > 0 else 0.0

    mav = float(np.mean(x_abs))
    rms = float(np.sqrt(np.mean(x_centered ** 2)))
    var = float(np.var(x_centered))
    wl = float(np.sum(np.abs(np.diff(x_centered))))

    zc = 0
    if x_centered.size > 1:
        zc = int(
            np.sum(
                ((x_centered[:-1] * x_centered[1:]) < 0) &
                (np.abs(np.diff(x_centered)) > thr)
            )
        )

    ssc = 0
    if x_centered.size > 2:
        dx1 = np.diff(x_centered)
        ssc = int(
            np.sum(
                ((dx1[:-1] * dx1[1:]) < 0) &
                (np.abs(dx1[:-1]) > thr) &
                (np.abs(dx1[1:]) > thr)
            )
        )

    wamp = int(np.sum(np.abs(np.diff(x_centered)) > thr)) if x_centered.size > 1 else 0
    iemg = float(np.sum(x_abs))

    return np.array([mav, rms, var, wl, zc, ssc, wamp, iemg], dtype=float)


def extract_imu_features(acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """Extract simple IMU features for a short window.

    - acc: array-like. Either shape (N,3) for x,y,z or (N,) for a magnitude sequence.
    - gyro: same shape semantics as acc.

    Returns concatenated features: [acc_mean, acc_std, acc_rms, acc_var, acc_max, acc_min,
                                     gyro_mean, gyro_std, gyro_rms, gyro_var, gyro_max, gyro_min]
    """
    def _axis_features(arr):
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return np.zeros(6, dtype=float)
        if a.ndim == 2 and a.shape[1] == 3:
            mag = np.linalg.norm(a, axis=1)
        else:
            mag = a.flatten()
        mean = float(np.mean(mag))
        std = float(np.std(mag))
        rms = float(np.sqrt(np.mean(mag ** 2)))
        var = float(np.var(mag))
        mx = float(np.max(mag))
        mn = float(np.min(mag))
        return np.array([mean, std, rms, var, mx, mn], dtype=float)

    acc_feats = _axis_features(acc)
    gyro_feats = _axis_features(gyro)
    return np.concatenate([acc_feats, gyro_feats])


# --- quick CLI test when run directly ---
if __name__ == '__main__':
    # small self-check
    emg_win = np.random.randn(89) * 0.02
    emg_win[10:15] += 0.5  # burst
    print('EMG features:', extract_emg_features(emg_win))

    acc = np.vstack([np.array([0.0, 0.0, 9.81]) + 0.1 * np.random.randn(3) for _ in range(50)])
    gyro = 0.01 * np.random.randn(50, 3)
    print('IMU features:', extract_imu_features(acc, gyro))
