import time
import math
import json
import serial
import numpy as np
import keyboard
import ctypes
import os
import sys

# ================= CONFIGURATION =================
SERIAL_PORT = 'COM10'
BAUD_RATE = 115200
SENSITIVITY_X = 800.0  
SENSITIVITY_Y = 800.0
SMOOTHING = 0.15

# ================= STATE MANAGEMENT =================
class ProgramState:
    def __init__(self):
        self.running = True
        self.paused = False
        self.recenter_requested = False
        
state = ProgramState()

# ================= HOTKEY CALLBACKS =================
# These run instantly when keys are pressed, regardless of the loop
def on_pause(e):
    state.paused = not state.paused
    status = "PAUSED (Mouse Free)" if state.paused else "ACTIVE (Tracking)"
    print(f"\r[{status}]                         ", end='')

def on_recenter(e):
    state.recenter_requested = True
    # We don't print here to avoid spamming console in-game

def on_kill(e):
    print("\n[EXIT] Kill switch activated.")
    state.running = False

# Register Hooks
keyboard.on_press_key('f8', on_pause)
keyboard.on_press_key('space', on_recenter)
keyboard.on_press_key('end', on_kill)

# ================= WINDOWS API =================
user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001

def move_mouse_relative(dx, dy):
    user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

# ================= MATH ENGINE =================
def quat_slerp(q1, q2, t):
    dot = np.dot(q1, q2)
    if dot < 0.0: q2 = -q2; dot = -dot
    if dot > 0.9995: return q1 + t * (q2 - q1)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q1) + (s1 * q2)

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_inverse(q): return np.array([q[0], -q[1], -q[2], -q[3]])

# ================= MAIN LOOP =================
def main():
    print("--- IMU GAME CONTROLLER ---")
    print(" [F8]    : Pause/Unpause")
    print(" [SPACE] : Recenter View")
    print(" [END]   : Kill Script")
    
    # 1. Admin Check
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("\n[WARNING] Not running as Admin. Hotkeys might fail in-game.\n")
    except: pass

    # 2. Load Calibration
    cal_yaw_axis = np.array([0.0, 1.0, 0.0])
    cal_pitch_axis = np.array([1.0, 0.0, 0.0])
    yaw_mult = 1.0
    pitch_mult = 1.0

    if os.path.exists('imu_calibration.json'):
        try:
            with open('imu_calibration.json', 'r') as f:
                config = json.load(f)
                cal_yaw_axis = np.array([config['yawAxis']['x'], config['yawAxis']['y'], config['yawAxis']['z']])
                cal_pitch_axis = np.array([config['pitchAxis']['x'], config['pitchAxis']['y'], config['pitchAxis']['z']])
                yaw_mult = config.get('yawMultiplier', 1.0)
                pitch_mult = config.get('pitchMultiplier', 1.0)
                print("[OK] Calibration Loaded.")
        except: print("[WARN] JSON Error. Using Defaults.")

    # 3. Connect Serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.005)
        ser.dtr = False; time.sleep(0.1); ser.dtr = True
        print("[OK] Connected.")
    except Exception as e:
        print(f"[FAIL] {e}")
        return

    # 4. State Variables
    current_quat = np.array([1.0, 0.0, 0.0, 0.0])
    tare_quat = np.array([1.0, 0.0, 0.0, 0.0])
    prev_yaw = 0.0
    prev_pitch = 0.0

    print(">>> ACTIVE <<<")

    while state.running:
        if ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('{') and '"quat"' in line:
                    data = json.loads(line)
                    q_raw = np.array(data['quat'])
                    q_raw /= np.linalg.norm(q_raw)

                    # Smooth
                    current_quat = quat_slerp(current_quat, q_raw, SMOOTHING)

                    # --- RECENTER EVENT ---
                    if state.recenter_requested:
                        tare_quat = current_quat.copy()
                        prev_yaw = 0.0
                        prev_pitch = 0.0
                        state.recenter_requested = False # Reset flag
                        continue

                    if state.paused:
                        continue

                    # Math: Calculate Rotation relative to Tare
                    q_delta = quat_multiply(quat_inverse(tare_quat), current_quat)
                    w = max(-1.0, min(1.0, q_delta[0]))
                    angle = 2 * math.acos(w)
                    
                    if angle < 0.00001: continue

                    s = math.sqrt(1 - w*w)
                    axis = q_delta[1:4] / s
                    
                    # Project to Axes
                    curr_yaw = angle * np.dot(axis, cal_yaw_axis) * yaw_mult
                    curr_pitch = angle * np.dot(axis, cal_pitch_axis) * pitch_mult

                    # Calculate Delta
                    delta_yaw = curr_yaw - prev_yaw
                    delta_pitch = curr_pitch - prev_pitch
                    
                    prev_yaw = curr_yaw
                    prev_pitch = curr_pitch
                    
                    # Move Mouse
                    dx = delta_yaw * SENSITIVITY_X
                    dy = -delta_pitch * SENSITIVITY_Y 
                    
                    if abs(dx) > 0.5 or abs(dy) > 0.5:
                        move_mouse_relative(dx, dy)
            except:
                pass
    
    # Clean exit
    ser.close()
    keyboard.unhook_all()
    print("\nTerminated.")

if __name__ == "__main__":
    main()