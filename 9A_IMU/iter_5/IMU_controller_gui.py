import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import serial
import serial.tools.list_ports
import json
import time
import math
import numpy as np
import threading
import keyboard
import ctypes
import os
import sys
import pyautogui

# ================= CONFIGURATION =================
# Default fallback settings
DEFAULT_COM = 'COM10'
BAUD_RATE = 115200

# ================= WINDOWS API (FAST INPUT) =================
user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001
SCREEN_W, SCREEN_H = pyautogui.size()

def move_mouse_relative(dx, dy):
    """Used for FPS Games (Delta Movement)"""
    user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

def move_mouse_absolute(x, y):
    """Used for Desktop (Pixel Mapping)"""
    # Clamp to screen
    x = max(0, min(SCREEN_W, x))
    y = max(0, min(SCREEN_H, y))
    # We use pyautogui for desktop as it handles multi-monitor better than raw SetCursorPos
    pyautogui.moveTo(x, y, _pause=False)

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

# ================= SHARED STATE =================
class AppState:
    def __init__(self):
        # Connectivity
        self.running = False
        self.paused = False
        self.serial_port = DEFAULT_COM
        
        # Settings (Live Tunable)
        self.mode = "DESKTOP" # or "GAME"
        self.sensitivity = 1.5
        self.smoothing = 0.15
        self.inv_yaw = False
        self.inv_pitch = False
        
        # Calibration Vectors
        self.cal_yaw_axis = np.array([0.0, 1.0, 0.0])
        self.cal_pitch_axis = np.array([1.0, 0.0, 0.0])
        self.yaw_mult = 1.0
        self.pitch_mult = 1.0
        
        # Tracking Data
        self.current_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.tare_quat = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Game Mode Deltas
        self.prev_yaw = 0.0
        self.prev_pitch = 0.0

state = AppState()

# ================= WORKER THREAD (THE ENGINE) =================
def engine_loop(gui_ref):
    # 1. Load Calibration
    try:
        if os.path.exists('imu_calibration.json'):
            with open('imu_calibration.json', 'r') as f:
                config = json.load(f)
                state.cal_yaw_axis = np.array([config['yawAxis']['x'], config['yawAxis']['y'], config['yawAxis']['z']])
                state.cal_pitch_axis = np.array([config['pitchAxis']['x'], config['pitchAxis']['y'], config['pitchAxis']['z']])
                # Normalize
                state.cal_yaw_axis /= np.linalg.norm(state.cal_yaw_axis)
                state.cal_pitch_axis /= np.linalg.norm(state.cal_pitch_axis)
                
                state.yaw_mult = config.get('yawMultiplier', 1.0)
                state.pitch_mult = config.get('pitchMultiplier', 1.0)
                gui_ref.log("Profile Loaded Successfully.")
        else:
            gui_ref.log("Warning: No JSON found. Using Defaults.")
    except Exception as e:
        gui_ref.log(f"Profile Error: {e}")

    # 2. Open Serial
    try:
        ser = serial.Serial(state.serial_port, BAUD_RATE, timeout=0.005)
        # Reset ESP32
        ser.dtr = False; time.sleep(0.1); ser.dtr = True
        gui_ref.update_status("ONLINE - ACTIVE", "success")
        gui_ref.log(f"Connected to {state.serial_port}")
    except Exception as e:
        gui_ref.log(f"Connection Failed: {e}")
        gui_ref.btn_toggle.configure(text="START ENGINE", bootstyle="success")
        state.running = False
        return

    # 3. Main Loop
    while state.running:
        if ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('{') and '"quat"' in line:
                    data = json.loads(line)
                    q_raw = np.array(data['quat'])
                    q_raw /= np.linalg.norm(q_raw)

                    # Smooth
                    state.current_quat = quat_slerp(state.current_quat, q_raw, state.smoothing)

                    if state.paused: 
                        continue

                    # Math: Delta from Tare
                    q_delta = quat_multiply(quat_inverse(state.tare_quat), state.current_quat)
                    w = max(-1.0, min(1.0, q_delta[0]))
                    angle = 2 * math.acos(w)
                    
                    if angle < 0.00001: continue

                    s = math.sqrt(1 - w*w)
                    axis = q_delta[1:4] / s
                    
                    # Apply User Inversions
                    y_dir = -1.0 if state.inv_yaw else 1.0
                    p_dir = -1.0 if state.inv_pitch else 1.0
                    
                    # Project
                    curr_yaw = angle * np.dot(axis, state.cal_yaw_axis) * state.yaw_mult * y_dir
                    curr_pitch = angle * np.dot(axis, state.cal_pitch_axis) * state.pitch_mult * p_dir

                    # --- MODE SWITCHING ---
                    if state.mode == "DESKTOP":
                        # ABSOLUTE MAPPING (Tan Projection)
                        x_off = math.tan(curr_yaw) * (SCREEN_H * state.sensitivity)
                        y_off = math.tan(curr_pitch) * (SCREEN_H * state.sensitivity)
                        
                        tx = (SCREEN_W / 2) - x_off
                        ty = (SCREEN_H / 2) - y_off
                        move_mouse_absolute(tx, ty)
                        
                    elif state.mode == "GAME":
                        # RELATIVE MAPPING (Delta Accumulation)
                        d_yaw = curr_yaw - state.prev_yaw
                        d_pitch = curr_pitch - state.prev_pitch
                        
                        # Update history
                        state.prev_yaw = curr_yaw
                        state.prev_pitch = curr_pitch
                        
                        # Scale for Game (High sensitivity needed for deltas)
                        dx = d_yaw * (state.sensitivity * 800.0) 
                        dy = -d_pitch * (state.sensitivity * 800.0)
                        
                        if abs(dx) > 0.5 or abs(dy) > 0.5:
                            move_mouse_relative(dx, dy)
                            
            except:
                pass
        else:
            time.sleep(0.001)

    ser.close()
    gui_ref.update_status("OFFLINE", "secondary")
    gui_ref.log("Engine Stopped.")

# ================= GUI CLASS =================
class ControlPanel(ttkb.Window):
    def __init__(self):
        super().__init__(themename="cyborg")
        self.title("IMU CONTROL CENTER")
        self.geometry("500x750")
        self.resizable(False, False)
        
        # Setup Hotkeys
        self.setup_hotkeys()
        self.create_ui()
        
    def setup_hotkeys(self):
        keyboard.on_press_key('f8', self.toggle_pause)
        keyboard.on_press_key('space', self.perform_recenter)
        keyboard.on_press_key('esc', self.kill_app)

    def toggle_pause(self, e):
        state.paused = not state.paused
        if state.paused:
            self.log("PAUSED (Mouse Free)")
            self.update_status("PAUSED", "warning")
        else:
            self.log("RESUMED (Tracking)")
            self.update_status("ACTIVE", "success")

    def perform_recenter(self, e):
        state.tare_quat = state.current_quat.copy()
        state.prev_yaw = 0.0
        state.prev_pitch = 0.0
        # self.log("Recenter (Tare)") # Commented out to reduce spam

    def kill_app(self, e):
        state.running = False
        self.destroy()
        sys.exit()

    def create_ui(self):
        # HEADER
        ttk.Label(self, text="IMU CONTROL CENTER", font=("Impact", 24), bootstyle="inverse-primary").pack(fill=X, ipady=10)
        
        self.lbl_status = ttk.Label(self, text="STATUS: READY", font=("Consolas", 12), bootstyle="secondary", anchor="center")
        self.lbl_status.pack(fill=X, pady=5)

        # MAIN FRAME
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=BOTH, expand=True)

        # 1. CONNECTION
        lf_conn = ttk.LabelFrame(main_frame, text="Connection", padding=15)
        lf_conn.pack(fill=X, pady=5)
        
        # Port Scanner
        self.cb_port = ttk.Combobox(lf_conn, values=self.get_ports())
        self.cb_port.set(DEFAULT_COM)
        self.cb_port.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))
        
        self.btn_refresh = ttk.Button(lf_conn, text="↻", command=self.refresh_ports, bootstyle="info-outline", width=3)
        self.btn_refresh.pack(side=LEFT, padx=(0, 10))
        
        self.btn_toggle = ttk.Button(lf_conn, text="START ENGINE", command=self.toggle_engine, bootstyle="success")
        self.btn_toggle.pack(side=RIGHT)

        # 2. OPERATION MODE
        lf_mode = ttk.LabelFrame(main_frame, text="Operation Mode", padding=15)
        lf_mode.pack(fill=X, pady=10)
        
        self.var_mode = tk.StringVar(value="DESKTOP")
        ttk.Radiobutton(lf_mode, text="DESKTOP (Absolute Cursor)", variable=self.var_mode, value="DESKTOP", command=self.update_vars, bootstyle="info-toolbutton").pack(side=LEFT, fill=X, expand=True, padx=5)
        ttk.Radiobutton(lf_mode, text="GAME (FPS Relative)", variable=self.var_mode, value="GAME", command=self.update_vars, bootstyle="danger-toolbutton").pack(side=LEFT, fill=X, expand=True, padx=5)

        # 3. TUNING
        lf_tune = ttk.LabelFrame(main_frame, text="Tuning", padding=15)
        lf_tune.pack(fill=X, pady=5)
        
        ttk.Label(lf_tune, text="Sensitivity (Zoom/Speed)", bootstyle="warning").pack(anchor=W)
        self.scale_sens = ttk.Scale(lf_tune, from_=0.1, to=5.0, command=self.update_vars)
        self.scale_sens.set(1.5)
        self.scale_sens.pack(fill=X, pady=(0, 15))
        
        ttk.Label(lf_tune, text="Smoothing (Left=Heavy, Right=Fast)", bootstyle="warning").pack(anchor=W)
        self.scale_smooth = ttk.Scale(lf_tune, from_=0.01, to=0.5, command=self.update_vars)
        self.scale_smooth.set(0.15)
        self.scale_smooth.pack(fill=X)

        # 4. AXIS CONTROL
        lf_axis = ttk.LabelFrame(main_frame, text="Axis Control", padding=15)
        lf_axis.pack(fill=X, pady=10)
        
        self.var_inv_y = tk.BooleanVar(value=False)
        self.var_inv_p = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(lf_axis, text="Invert Yaw (Left/Right)", variable=self.var_inv_y, command=self.update_vars, bootstyle="round-toggle").pack(side=LEFT, expand=True)
        ttk.Checkbutton(lf_axis, text="Invert Pitch (Up/Down)", variable=self.var_inv_p, command=self.update_vars, bootstyle="round-toggle").pack(side=LEFT, expand=True)

        # 5. ACTIONS
        ttk.Button(main_frame, text="RECENTER VIEW (SPACE)", command=lambda: self.perform_recenter(None), bootstyle="primary-outline").pack(fill=X, pady=20, ipady=5)

        # 6. LOG
        self.txt_log = tk.Text(main_frame, height=8, font=("Consolas", 9), bg="#111", fg="#0f0", state=DISABLED)
        self.txt_log.pack(fill=X)
        
        self.log("Ready. Select COM Port and Start.")

    # --- LOGIC ---
    def get_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def refresh_ports(self):
        self.cb_port['values'] = self.get_ports()
        
    def update_vars(self, _=None):
        state.sensitivity = self.scale_sens.get()
        state.smoothing = self.scale_smooth.get()
        state.mode = self.var_mode.get()
        state.inv_yaw = self.var_inv_y.get()
        state.inv_pitch = self.var_inv_p.get()

    def toggle_engine(self):
        if not state.running:
            state.serial_port = self.cb_port.get()
            state.running = True
            self.btn_toggle.configure(text="STOP ENGINE", bootstyle="danger")
            self.cb_port.configure(state=DISABLED)
            
            # Start Thread
            t = threading.Thread(target=engine_loop, args=(self,))
            t.daemon = True
            t.start()
        else:
            state.running = False
            self.btn_toggle.configure(text="START ENGINE", bootstyle="success")
            self.cb_port.configure(state=NORMAL)

    def log(self, msg):
        self.txt_log.configure(state=NORMAL)
        self.txt_log.insert(tk.END, f"> {msg}\n")
        self.txt_log.see(tk.END)
        self.txt_log.configure(state=DISABLED)

    def update_status(self, text, style):
        self.lbl_status.configure(text=f"STATUS: {text}", bootstyle=style)

if __name__ == "__main__":
    # Check Admin
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("WARNING: Not running as Admin. Hotkeys may fail in-game.")
    except: pass

    app = ControlPanel()
    app.mainloop()