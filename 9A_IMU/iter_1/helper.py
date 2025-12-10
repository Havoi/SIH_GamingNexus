#!/usr/bin/env python3
"""
imu_map_debug.py

Quick diagnostic & auto-map for IMU -> screen mapping.

Usage:
  1) Live serial:
       python imu_map_debug.py --serial COM3 --baud 115200
  2) From file of newline JSON lines:
       python imu_map_debug.py --file sample_lines.txt

The script will guide you:
 - Hold REST (still)
 - Rotate RIGHT (hold), LEFT, UP, DOWN when prompted.
It prints the recommended mapping (which Euler -> X/Y and invert flags).
It also writes calibration.json if --save is provided.
"""

import json, time, argparse, math, sys
from statistics import mean
from collections import deque
import numpy as np

# ------------------------------
# Quaternion helpers (numpy)
# ------------------------------
def normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0,0.0,0.0,0.0])
    return q / n

def quat_conj(q):
    q = np.array(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def quat_mul(a,b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_to_euler_zyx(q):
    # returns roll,pitch,yaw (radians)
    w,x,y,z = q
    sinr = 2*(w*x + y*z); cosr = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = 2*(w*y - z*x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)
    else:
        pitch = math.asin(sinp)
    siny = 2*(w*z + x*y); cosy = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw

# ------------------------------
# Data source helpers
# ------------------------------
class FileSource:
    def __init__(self, path):
        self.f = open(path, 'r', encoding='utf8')
    def read_line(self, timeout=0.1):
        # return next non-empty line or None
        while True:
            line = self.f.readline()
            if not line:
                return None
            s = line.strip()
            if s:
                return s

class SerialSource:
    def __init__(self, port, baud=115200, timeout=0.1):
        import serial
        self.ser = serial.Serial(port, baud, timeout=timeout)
    def read_line(self, timeout=0.1):
        try:
            line = self.ser.readline().decode('utf8', errors='ignore')
            if line:
                return line.strip()
        except Exception as e:
            return None

# ------------------------------
# Capture routine
# ------------------------------
def collect_samples(source, duration_s=0.8, label="CAPTURE"):
    print(f"\n--- {label} : wait 1.2s and then hold the pose for {duration_s:.1f}s ---")
    time.sleep(1.2)
    print(f"  Capturing {label} now...")
    samples = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        line = source.read_line()
        if not line:
            time.sleep(0.005); continue
        try:
            obj = json.loads(line)
        except:
            continue
        if isinstance(obj, dict) and obj.get("type") == "imu" and "quat" in obj:
            q = obj["quat"][:4]
            # Heuristic: if incoming looks like [x,y,z,w] fix ordering
            if abs(q[0]) < 0.2 and abs(q[3]) > 0.7:
                q = [q[3], q[0], q[1], q[2]]
            qn = normalize(q)
            samples.append(qn)
    print(f"  Captured {len(samples)} samples for {label}")
    return samples

def mean_quat(quats):
    # simple Markley average using numpy
    X = np.zeros((4,4))
    for q in quats:
        v = q.reshape(4,1)
        X += v @ v.T
    X /= len(quats)
    vals, vecs = np.linalg.eigh(X)
    q = vecs[:, np.argmax(vals)]
    return normalize(q.real)

# ------------------------------
# Auto mapping logic
# ------------------------------
def analyze_mapping(rest, right, left, up, down, baseline=None):
    # If baseline provided, convert samples relative to baseline: q_rel = inv(baseline)*q
    def relify(samples):
        if baseline is None:
            return samples
        invb = quat_conj(baseline)
        out = []
        for q in samples:
            qr = quat_mul(invb, q)
            out.append(normalize(qr))
        return out

    rest_r = mean_quat(relify(rest))
    r_r = mean_quat(relify(right))
    l_r = mean_quat(relify(left))
    u_r = mean_quat(relify(up))
    d_r = mean_quat(relify(down))

    rest_e = np.array(quat_to_euler_zyx(rest_r))
    r_e = np.array(quat_to_euler_zyx(r_r))
    l_e = np.array(quat_to_euler_zyx(l_r))
    u_e = np.array(quat_to_euler_zyx(u_r))
    d_e = np.array(quat_to_euler_zyx(d_r))

    # deltas magnitudes (absolute)
    horiz = (np.abs(r_e - rest_e) + np.abs(l_e - rest_e)) / 2.0
    vert  = (np.abs(u_e - rest_e) + np.abs(d_e - rest_e)) / 2.0

    # pick dominant axis indices (0=roll,1=pitch,2=yaw)
    horiz_idx = int(np.argmax(horiz))
    vert_idx = int(np.argmax(vert))
    if horiz_idx == vert_idx:
        # if same, pick next best for vert
        vert_idx = int(np.argsort(vert)[-2]) if np.argsort(vert)[-1] == horiz_idx else int(np.argsort(vert)[-1])

    idx2name = {0:'roll',1:'pitch',2:'yaw'}
    xaxis = idx2name[horiz_idx]; yaxis = idx2name[vert_idx]

    # sign: check direction of change
    right_dir = (r_e - rest_e)[horiz_idx]
    up_dir = (u_e - rest_e)[vert_idx]
    inv_x = True if right_dir < 0 else False
    inv_y = True if up_dir < 0 else False

    # produce numeric diagnostics
    diag = {
        'rest_e_deg': (rest_e*180/math.pi).tolist(),
        'right_e_deg': (r_e*180/math.pi).tolist(),
        'left_e_deg': (l_e*180/math.pi).tolist(),
        'up_e_deg': (u_e*180/math.pi).tolist(),
        'down_e_deg': (d_e*180/math.pi).tolist(),
        'horiz_mag_deg': (horiz*180/math.pi).tolist(),
        'vert_mag_deg': (vert*180/math.pi).tolist()
    }

    mapping = {'xAxis': xaxis, 'yAxis': yaxis, 'invX': inv_x, 'invY': inv_y}
    return mapping, diag

# ------------------------------
# CLI & flow
# ------------------------------
def run_flow(source, save=False):
    print("== IMU Auto-Map Debugger ==")
    print("Make sure IMU packets look like: {\"type\":\"imu\",\"quat\":[w,x,y,z],...}")
    input("Press Enter when ready to start REST -> RIGHT -> LEFT -> UP -> DOWN captures (you will be prompted)...")

    rest = collect_samples(source, duration_s=0.8, label="REST (hold still)")
    if len(rest) < 8:
        print("Too few rest samples. Is IMU streaming? Aborting.")
        return
    right = collect_samples(source, duration_s=0.8, label="ROTATE RIGHT (hold)")
    left  = collect_samples(source, duration_s=0.8, label="ROTATE LEFT (hold)")
    up    = collect_samples(source, duration_s=0.8, label="ROTATE UP (pitch up, hold)")
    down  = collect_samples(source, duration_s=0.8, label="ROTATE DOWN (pitch down, hold)")

    print("\nComputing mapping...")
    mapping, diag = analyze_mapping(rest, right, left, up, down, baseline=None)
    print("\n=== Recommended mapping (paste into web UI) ===")
    print(json.dumps(mapping, indent=2))
    print("\n=== Diagnostic (degrees) ===")
    for k,v in diag.items():
        print(f"{k}: {['{:.2f}'.format(x) for x in v]}")

    if save:
        # baseline from rest average
        baseline = mean_quat(rest)
        out = {
            'baseline_quaternion': baseline.tolist(),
            'mapping': mapping,
            'diag': diag,
            'meta': {'generated': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        }
        with open('calibration_auto_debug.json','w') as f:
            json.dump(out, f, indent=2)
        print("\nSaved calibration_auto_debug.json")

    print("\nIf mapping looks wrong, run again or inspect 'diag' values above. Large horiz_mag on axis -> that axis maps horizontal.")
    print("To apply mapping in the web UI, copy the JSON mapping object and set window.__imuMapping = <that object>")
    print("Example (paste into browser console):")
    print("  window.__imuMapping = " + json.dumps(mapping))
    print("Then in console test mapping by moving IMU; to invert flips you can do:")
    print("  window.__imuMapping.invX = !window.__imuMapping.invX")
    print("  window.__imuMapping.invY = !window.__imuMapping.invY")
    print("\nDone.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--serial', help='Serial port (e.g. COM3 or /dev/ttyUSB0)')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--file', help='Read newline JSON lines from file (instead of serial)')
    p.add_argument('--save', action='store_true', help='Save calibration_auto_debug.json')
    args = p.parse_args()

    if not args.serial and not args.file:
        print("Must provide --serial or --file")
        sys.exit(1)

    if args.file:
        source = FileSource(args.file)
    else:
        source = SerialSource(args.serial, args.baud)

    try:
        run_flow(source, save=args.save)
    except KeyboardInterrupt:
        print("Aborted by user")
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()
