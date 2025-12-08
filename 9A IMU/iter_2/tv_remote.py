#!/usr/bin/env python3
"""
imu_tvremote_mouse.py
Single-file IMU -> TV-remote-like mouse controller.

Usage:
  1) Edit PORT or provide CLI override.
  2) Run: python imu_tvremote_mouse.py
  3) Use keys in console:
     c = REST calibrate (hold IMU still)
     m = Auto-map (guided: RIGHT, LEFT, UP, DOWN)
     r = Quick recenter (set session baseline from last sample)
     v = Save calibration.json
     l = Load calibration.json
     SPACE = toggle mouse on/off
     s / a = sensitivity up / down
     f / d = smoothing up / down
     Esc = exit

Input format expected on serial: newline-delimited JSON lines like:
 {"type":"imu","quat":[w,x,y,z],"acc":[...],"gyro":[...],"t":...}
The script will attempt to auto-correct [x,y,z,w] ordering heuristically.

Outputs:
 - calibration.json saved with baseline_quaternion and mapping.
"""

import json, time, math, os, sys, threading
from statistics import mean
from collections import deque
import numpy as np
import serial
import pyautogui
from pynput import keyboard

# -----------------------
# CONFIG - change if needed
# -----------------------
PORT = "COM10"          # <-- change to your serial port or pass via CLI (not implemented here)
BAUD = 115200
SAMPLES_REST = 300     # samples to average for REST baseline
CAPTURE_MS = 800       # ms for each gesture capture in auto-map
SAVE_FILE = "calibration.json"
# Mouse scaling defaults
PIXELS_PER_UNIT = 1200.0  # pixels per normalized unit per second (tune later)
MAX_STEP = 400             # clamp per-frame pixel move
pyautogui.FAILSAFE = False

# -----------------------
# Quaternion helpers
# -----------------------
def normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def quat_mul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_to_euler_zyx(q):
    w,x,y,z = q
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = 2*(w*y - z*x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)
    else:
        pitch = math.asin(sinp)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw  # radians

# -----------------------
# Serial reader (thread)
# -----------------------
class SerialReader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self._stop = threading.Event()
        self._ser = None
        self.last_line = None
        self.q_latest = None

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
            print(f"[Serial] Opened {self.port} @ {self.baud}")
        except Exception as e:
            print("[Serial] Open error:", e)
            return
        buf = b''
        while not self._stop.is_set():
            try:
                b = self._ser.read(1024)
                if b:
                    buf += b
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        s = line.decode('utf8', errors='ignore').strip()
                        if s:
                            self.last_line = s
                            self._handle_line(s)
                else:
                    time.sleep(0.002)
            except Exception as e:
                print("[Serial] Read error:", e)
                time.sleep(0.1)
        try:
            self._ser.close()
        except:
            pass
        print("[Serial] Stopped")

    def _handle_line(self, line):
        try:
            obj = json.loads(line)
        except:
            return
        if isinstance(obj, dict) and obj.get("type") == "imu" and "quat" in obj:
            q = obj["quat"][:4]
            # heuristic: if q[0] small and q[3] close to 1 -> assume MCU sent [x,y,z,w]
            if abs(q[0]) < 0.2 and abs(q[3]) > 0.7:
                q = [q[3], q[0], q[1], q[2]]
            self.q_latest = normalize(q)

    def stop(self):
        self._stop.set()

# -----------------------
# Auto-map & calibration
# -----------------------
def avg_quaternion(quats):
    # Markley method
    M = np.zeros((4,4))
    for q in quats:
        v = q.reshape(4,1)
        M += v @ v.T
    M /= len(quats)
    vals, vecs = np.linalg.eigh(M)
    q = vecs[:, np.argmax(vals)]
    return normalize(q.real)

def collect_samples(reader, ms=CAPTURE_MS):
    # collect q samples for duration ms
    samples = []
    t0 = time.time()
    while (time.time() - t0) * 1000.0 < ms:
        q = reader.q_latest
        if q is not None:
            samples.append(q.copy())
        time.sleep(0.001)
    return samples

def rest_calibrate(reader, n=SAMPLES_REST):
    print("[CAL] Rest calibration — keep IMU perfectly still for a few seconds...")
    samples = []
    t0 = time.time()
    while len(samples) < n and (time.time() - t0) < max(5, n/200.0 + 2):
        q = reader.q_latest
        if q is not None:
            samples.append(q.copy())
        time.sleep(0.002)
    if len(samples) < 8:
        print("[CAL] Too few samples for rest calibration.")
        return None
    qavg = avg_quaternion(np.array(samples))
    print("[CAL] Baseline saved.")
    return qavg

def auto_map(reader, baseline=None):
    print("\n[AUTOMAP] Follow prompts. Keep IMU steady for REST then perform gestures.")
    input("Press Enter to start REST capture (hold still)...")
    rest = collect_samples(reader, ms=700)
    print(f"[AUTOMAP] REST samples: {len(rest)}")
    input("Press Enter and then rotate RIGHT (face turning right) and hold...")
    right = collect_samples(reader, ms=700)
    print(f"[AUTOMAP] RIGHT samples: {len(right)}")
    input("Press Enter and then rotate LEFT and hold...")
    left = collect_samples(reader, ms=700)
    print(f"[AUTOMAP] LEFT samples: {len(left)}")
    input("Press Enter and then PITCH UP (tilt away) and hold...")
    up = collect_samples(reader, ms=700)
    print(f"[AUTOMAP] UP samples: {len(up)}")
    input("Press Enter and then PITCH DOWN (tilt towards) and hold...")
    down = collect_samples(reader, ms=700)
    print(f"[AUTOMAP] DOWN samples: {len(down)}")

    # if baseline provided, make relative
    def rel(samples):
        if baseline is None:
            return samples
        invb = quat_conj(baseline)
        out = []
        for q in samples:
            out.append(normalize(quat_mul(invb, q)))
        return out

    # compute means
    try:
        rest_avg = avg_quaternion(np.array(rel(rest)))
        r_avg = avg_quaternion(np.array(rel(right)))
        l_avg = avg_quaternion(np.array(rel(left)))
        u_avg = avg_quaternion(np.array(rel(up)))
        d_avg = avg_quaternion(np.array(rel(down)))
    except Exception as e:
        print("[AUTOMAP] error computing averages:", e)
        return None

    rest_e = np.array(quat_to_euler_zyx(rest_avg))
    r_e = np.array(quat_to_euler_zyx(r_avg))
    l_e = np.array(quat_to_euler_zyx(l_avg))
    u_e = np.array(quat_to_euler_zyx(u_avg))
    d_e = np.array(quat_to_euler_zyx(d_avg))

    horiz = (np.abs(r_e - rest_e) + np.abs(l_e - rest_e)) / 2.0
    vert  = (np.abs(u_e - rest_e) + np.abs(d_e - rest_e)) / 2.0

    horiz_idx = int(np.argmax(horiz))
    vert_idx = int(np.argmax(vert))
    if horiz_idx == vert_idx:
        # choose second best for vertical
        sorted_idx = np.argsort(vert)[::-1]
        vert_idx = int(sorted_idx[0] if sorted_idx[0] != horiz_idx else sorted_idx[1])

    idx2name = {0:'roll', 1:'pitch', 2:'yaw'}
    xAxis = idx2name[horiz_idx]; yAxis = idx2name[vert_idx]
    right_dir = (r_e - rest_e)[horiz_idx]
    up_dir = (u_e - rest_e)[vert_idx]
    invX = True if right_dir < 0 else False
    invY = True if up_dir < 0 else False

    mapping = {'xAxis': xAxis, 'yAxis': yAxis, 'invX': invX, 'invY': invY}
    diag = {
        'rest_e_deg': (rest_e*180/math.pi).tolist(),
        'right_e_deg': (r_e*180/math.pi).tolist(),
        'left_e_deg': (l_e*180/math.pi).tolist(),
        'up_e_deg': (u_e*180/math.pi).tolist(),
        'down_e_deg': (d_e*180/math.pi).tolist(),
        'horiz_mag_deg': (horiz*180/math.pi).tolist(),
        'vert_mag_deg': (vert*180/math.pi).tolist()
    }
    print("\n[AUTOMAP] Recommended mapping:", mapping)
    print("[AUTOMAP] Diagnostic (degrees):")
    for k,v in diag.items():
        print(f"  {k}: {[f'{x:.1f}' for x in v]}")
    return mapping

# -----------------------
# Main mouse control pipeline
# -----------------------
class TVMouseController:
    def __init__(self):
        self.baseline = None      # q0
        self.mapping = None       # dict with xAxis, yAxis, invX, invY
        self.sensitivity = 1.0    # multiplier
        self.smooth = 0.88        # EMA alpha (0..0.98) higher = more smoothing
        self.deadzone_deg = 1.5
        self.enabled = False
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.last_time = None

    def set_calibration(self, q0, mapping):
        self.baseline = q0.copy() if q0 is not None else None
        self.mapping = mapping.copy() if mapping is not None else None
        print("[CTRL] Calibration loaded:", "baseline" if q0 is not None else "no baseline", mapping)

    def process_and_move(self, q_raw):
        # q_raw is [w,x,y,z] numpy
        if q_raw is None or self.mapping is None:
            return
        # relative to baseline
        if self.baseline is not None:
            q_rel = quat_mul(quat_conj(self.baseline), q_raw)
            q_rel = normalize(q_rel)
        else:
            q_rel = q_raw

        roll, pitch, yaw = quat_to_euler_zyx(q_rel)
        axis_vals = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        vx = axis_vals[self.mapping['xAxis']]
        vy = axis_vals[self.mapping['yAxis']]
        if self.mapping.get('invX', False): vx = -vx
        if self.mapping.get('invY', False): vy = -vy

        # deadzone
        dz = math.radians(self.deadzone_deg)
        if abs(vx) < dz: vx = 0.0
        if abs(vy) < dz: vy = 0.0

        # sensitivity mapping: convert radians to a bounded normalized -1..1 using tanh curve
        # tune sensitivity (higher => more movement for same angle)
        nx = math.tanh(vx * self.sensitivity)
        ny = math.tanh(vy * self.sensitivity)

        # smoothing (EMA on normalized values)
        a = self.smooth
        if not hasattr(self, 'smooth_init') or not self.smooth_init:
            self.filtered_x = nx; self.filtered_y = ny; self.smooth_init = True
        else:
            self.filtered_x = a * self.filtered_x + (1 - a) * nx
            self.filtered_y = a * self.filtered_y + (1 - a) * ny

        # convert to pixel movement per frame based on time delta
        now = time.time()
        if self.last_time is None:
            dt = 1/60.0
        else:
            dt = max(1e-4, now - self.last_time)
        self.last_time = now

        # Pixels step proportional to filtered normalized value and sensitivity factor
        px = self.filtered_x * PIXELS_PER_UNIT * dt
        py = -self.filtered_y * PIXELS_PER_UNIT * dt  # invert Y for screen coordinates

        # clamp
        px = max(-MAX_STEP, min(MAX_STEP, px))
        py = max(-MAX_STEP, min(MAX_STEP, py))

        # apply move
        if self.enabled and (abs(px) >= 0.5 or abs(py) >= 0.5):
            try:
                pyautogui.moveRel(px, py, duration=0)
            except Exception as e:
                # ignore move errors
                print("[MOUSE] move error:", e)

        # telemetry print (compact)
        sys.stdout.write(f"\rroll={math.degrees(roll):6.1f} pitch={math.degrees(pitch):6.1f} yaw={math.degrees(yaw):6.1f} | nx={self.filtered_x:+.3f} ny={self.filtered_y:+.3f} enabled={self.enabled} ")
        sys.stdout.flush()

# -----------------------
# Save/Load calibration JSON
# -----------------------
def save_calibration(q0, mapping, filename=SAVE_FILE):
    out = {
        "baseline_quaternion": q0.tolist() if q0 is not None else None,
        "mapping": mapping,
        "meta": {"created": time.strftime("%Y-%m-%d %H:%M:%S")}
    }
    with open(filename, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] calibration written to {filename}")

def load_calibration(filename=SAVE_FILE):
    if not os.path.exists(filename):
        print("[LOAD] no file:", filename); return None, None
    with open(filename, "r") as f:
        js = json.load(f)
    q0 = np.array(js.get("baseline_quaternion"), dtype=float) if js.get("baseline_quaternion") else None
    mapping = js.get("mapping")
    print("[LOAD] loaded", filename, "mapping:", mapping)
    return q0, mapping

# -----------------------
# Keyboard controls (global)
# -----------------------
controller = TVMouseController()
reader = None
serial_thread = None
running = True

def on_press(key):
    global running, serial_thread, reader, controller
    try:
        k = key.char
    except AttributeError:
        if key == keyboard.Key.space:
            controller.enabled = not controller.enabled
            print("\n[MOUSE] enabled =", controller.enabled)
            return
        if key == keyboard.Key.esc:
            print("\n[EXIT] Exiting...")
            running = False
            return
        return

    if k == 'c':  # rest calibrate
        q0 = rest_calibrate(reader)
        if q0 is not None:
            controller.baseline = q0
            print("[KEY] baseline set")
    elif k == 'm':  # automap
        mapping = auto_map(reader, baseline=controller.baseline)
        if mapping:
            controller.mapping = mapping
            print("[KEY] mapping set to:", mapping)
    elif k == 'r':  # quick recenter
        if reader.q_latest is not None:
            controller.baseline = reader.q_latest.copy()
            print("[KEY] quick recenter set")
    elif k == 'v':
        if controller.baseline is not None and controller.mapping is not None:
            save_calibration(controller.baseline, controller.mapping)
        else:
            print("[KEY] need baseline + mapping to save")
    elif k == 'l':
        q0, mapping = load_calibration()
        if mapping:
            controller.set_calibration(q0, mapping)
    elif k == 's':
        controller.sensitivity *= 1.1
        print(f"[KEY] sensitivity -> {controller.sensitivity:.3f}")
    elif k == 'a':
        controller.sensitivity /= 1.1
        print(f"[KEY] sensitivity -> {controller.sensitivity:.3f}")
    elif k == 'f':
        controller.smooth = min(0.99, controller.smooth + 0.02)
        print(f"[KEY] smooth -> {controller.smooth:.3f}")
    elif k == 'd':
        controller.smooth = max(0.0, controller.smooth - 0.02)
        print(f"[KEY] smooth -> {controller.smooth:.3f}")
    elif k == 'h':
        print_help()
    else:
        pass

def print_help():
    print("""
Keys:
  c = REST calibrate (hold IMU still)
  m = AUTO-MAP (guided gestures: RIGHT, LEFT, UP, DOWN)
  r = Quick recenter (use last sample)
  v = Save calibration.json
  l = Load calibration.json
  SPACE = Toggle mouse ON/OFF
  s / a = sensitivity up / down
  f / d = smoothing up / down
  h = help
  Esc = exit
""")

# -----------------------
# Main
# -----------------------
def main():
    global reader, serial_thread, controller, running
    print("IMU TV-Remote Mouse Controller")
    print("Open serial port:", PORT)
    reader = SerialReader(PORT, BAUD)
    reader.start()
    time.sleep(0.5)

    # try to load calibration if present
    q0, mapping = load_calibration()
    if mapping:
        controller.set_calibration(q0, mapping)

    # keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print_help()
    try:
        while running:
            q = reader.q_latest
            controller.process_and_move(q)
            time.sleep(0.005)  # a tiny sleep, loop runs ~200Hz if possible
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        reader.stop()
        listener.stop()
        print("\nShutdown complete.")

if __name__ == "__main__":
    main()
