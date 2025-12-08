#!/usr/bin/env python3
"""
imu_gestures.py
Simple gesture detector for IMU JSON stream over serial.

Reads lines like:
 {"type":"imu","quat":[w,x,y,z],"acc":[ax,ay,az],"gyro":[gx,gy,gz],"t":4367}

Usage:
  pip install pyserial numpy
  python imu_gestures.py --port /dev/ttyUSB0 --baud 115200

Tweak thresholds at the top of the file if needed.
"""
import argparse
import json
import time
from collections import deque
from math import atan2, asin, sqrt, degrees
import numpy as np
import serial

# ---------------- CONFIG (tweak if necessary) ----------------
WINDOW_MS = 800                # sliding analysis window
GESTURE_COOLDOWN_MS = 600
FLICK_THRESH_DPS = 180         # deg/s for flick detection
ROLL_FLICK_THRESH = 240
SHAKE_AVG_GYRO = 120          # deg/s average to consider a shake
SHAKE_MIN_FLIPS = 2
CIRCLE_DEG = 200              # degrees yaw change to count as circle
HOLD_MAX_GYRO = 8            # max gyro deg/s for hold detection
HOLD_MS = 650                # how long stable to be a hold
DOUBLE_TAP_MAX_MS = 500      # double tap timing

# ---------------- utilities ----------------
def quat_to_euler_deg(qw, qx, qy, qz):
    """
    Convert quaternion (w,x,y,z) to Euler angles in degrees.
    Uses the following convention:
      roll  = rotation about X axis
      pitch = rotation about Y axis
      yaw   = rotation about Z axis
    Reference formulas (safe for numerical edge cases).
    """
    # roll (x-axis rotation)
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = max(-1.0, min(1.0, t2))
    pitch = asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = atan2(t3, t4)

    return degrees(yaw), degrees(pitch), degrees(roll)

# ---------------- sample buffer ----------------
class SampleBuffer:
    def __init__(self, window_ms):
        self.window_ms = window_ms
        self.deque = deque()

    def push(self, entry):
        """entry: dict with keys t (ms), quat [w,x,y,z], gyro [gx,gy,gz], acc [ax,ay,az]"""
        self.deque.append(entry)
        self._expire_old()

    def _expire_old(self):
        now = self.deque[-1]['t']
        while self.deque and (now - self.deque[0]['t']) > self.window_ms:
            self.deque.popleft()

    def get_list(self):
        return list(self.deque)

    def is_enough(self, min_count=3):
        return len(self.deque) >= min_count

# ---------------- gesture detectors ----------------
class GestureDetector:
    def __init__(self):
        self.buf = SampleBuffer(WINDOW_MS)
        self.last_detect = 0
        self.last_tap_time = 0
        self.last_shake = 0
        self.hold_start = None

    def push_sample(self, sample):
        self.buf.push(sample)
        # update last gyro for other systems if needed
        self.detect_all()

    def now_ms(self):
        return int(time.time() * 1000)

    def cooldown_ok(self):
        return (self.now_ms() - self.last_detect) > GESTURE_COOLDOWN_MS

    def detect_all(self):
        if not self.buf.is_enough():
            return
        if not self.cooldown_ok():
            return

        if self.detect_flick(): return
        if self.detect_double_tap(): return
        if self.detect_shake(): return
        if self.detect_circle(): return
        if self.detect_hold(): return

    def compute_ang_vel(self):
        """
        Approximate angular velocity (deg/s) by computing difference between
        first and last Euler angles in the buffer and dividing by dt.
        Returns dict: {'yaw':..., 'pitch':..., 'roll':...}
        """
        lst = self.buf.get_list()
        q0 = lst[0]['quat']; q1 = lst[-1]['quat']
        t0 = lst[0]['t'] / 1000.0
        t1 = lst[-1]['t'] / 1000.0
        dt = t1 - t0
        if dt <= 0.0001:
            return {'yaw':0.0, 'pitch':0.0, 'roll':0.0}

        y0, p0, r0 = quat_to_euler_deg(*q0)
        y1, p1, r1 = quat_to_euler_deg(*q1)

        # unwrap yaw/pitch/roll differences to nearest angle
        def angle_diff(a2, a1):
            d = a2 - a1
            while d > 180: d -= 360
            while d < -180: d += 360
            return d

        dy = angle_diff(y1, y0)
        dp = angle_diff(p1, p0)
        dr = angle_diff(r1, r0)
        return {'yaw': dy / dt, 'pitch': dp / dt, 'roll': dr / dt}

    def detect_flick(self):
        ang = self.compute_ang_vel()
        abs_yaw = abs(ang['yaw'])
        abs_pitch = abs(ang['pitch'])
        abs_roll = abs(ang['roll'])
        if abs_yaw > FLICK_THRESH_DPS:
            self.emit('Flick Yaw', f"{abs_yaw:.0f}°/s")
            return True
        if abs_pitch > FLICK_THRESH_DPS:
            self.emit('Flick Pitch', f"{abs_pitch:.0f}°/s")
            return True
        if abs_roll > ROLL_FLICK_THRESH:
            self.emit('Flick Roll', f"{abs_roll:.0f}°/s")
            return True
        return False

    def detect_double_tap(self):
        # quick pitch bounce: look for a significant pitch delta in buffer and check last tap timing
        lst = self.buf.get_list()
        pitches = [quat_to_euler_deg(*s['quat'])[1] for s in lst]  # pitch is index 1
        if len(pitches) < 4:
            return False
        diffs = np.diff(pitches)
        maxdiff = np.max(np.abs(diffs))
        if maxdiff > 12:  # quick bounce threshold (deg)
            now = self.now_ms()
            if (now - self.last_tap_time) < DOUBLE_TAP_MAX_MS:
                self.last_tap_time = 0
                self.emit('Double Tap (pitch)')
                return True
            else:
                self.last_tap_time = now
        return False

    def detect_shake(self):
        # use gyro magnitudes and sign flips
        lst = self.buf.get_list()
        mags = [sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]) for g in (s['gyro'] for s in lst)]
        avg = np.mean(mags) if mags else 0
        if avg > SHAKE_AVG_GYRO:
            # count sign flips in gyro X across buffer
            flips = 0
            prev = lst[0]['gyro'][0]
            for s in lst[1:]:
                cur = s['gyro'][0]
                if prev * cur < 0:
                    flips += 1
                prev = cur
            if flips >= SHAKE_MIN_FLIPS and (self.now_ms() - self.last_shake) > 800:
                self.last_shake = self.now_ms()
                self.emit('Shake')
                return True
        return False

    def detect_circle(self):
        # approximate yaw sweep over window > CIRCLE_DEG
        lst = self.buf.get_list()
        yaws = [quat_to_euler_deg(*s['quat'])[0] for s in lst]
        if len(yaws) < 6:
            return False
        # unwrap
        unwrapped = [yaws[0]]
        for i in range(1, len(yaws)):
            d = yaws[i] - yaws[i-1]
            if d > 180: d -= 360
            if d < -180: d += 360
            unwrapped.append(unwrapped[-1] + d)
        delta = abs(unwrapped[-1] - unwrapped[0])
        if delta > CIRCLE_DEG:
            self.emit('Circle', f"{delta:.0f}°")
            return True
        return False

    def detect_hold(self):
        lst = self.buf.get_list()
        mags = [sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]) for g in (s['gyro'] for s in lst)]
        maxmag = max(mags) if mags else 0
        if maxmag < HOLD_MAX_GYRO:
            # either continue hold or start
            oldest_t = lst[0]['t']
            # hold condition: window length >= HOLD_MS
            if (lst[-1]['t'] - oldest_t) > HOLD_MS:
                self.emit('Hold')
                return True
        return False

    def emit(self, name, info=None):
        self.last_detect = self.now_ms()
        now_s = time.strftime("%H:%M:%S")
        if info:
            print(f"[{now_s}] Gesture: {name}  ({info})")
        else:
            print(f"[{now_s}] Gesture: {name}")

# ---------------- serial reading ----------------
def run_serial_reader(port_name, baud):
    ser = serial.Serial(port_name, baud, timeout=1)
    detector = GestureDetector()
    print("Connected to", port_name, "at", baud)
    try:
        partial = ""
        while True:
            raw = ser.readline().decode(errors='ignore')
            if not raw:
                # no data this iteration; small sleep
                time.sleep(0.002)
                continue
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # ignore non-json lines
                continue
            if obj.get('type') != 'imu' or 'quat' not in obj:
                continue
            # normalize data
            quat = obj['quat']  # [w,x,y,z]
            gyro = obj.get('gyro', [0,0,0])
            acc = obj.get('acc', [0,0,0])
            sample = {
                't': int(time.time() * 1000),
                'quat': quat,
                'gyro': gyro,
                'acc': acc
            }
            detector.push_sample(sample)
    finally:
        ser.close()

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser(description="IMU gesture detector (serial JSON input).")
    p.add_argument("--port", required=True, help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    args = p.parse_args()
    run_serial_reader(args.port, args.baud)

if __name__ == "__main__":
    main()
