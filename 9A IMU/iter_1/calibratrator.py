#!/usr/bin/env python3
"""
IMU Global Axis Calibrator (PyQt5)
- Connects to serial port streaming newline JSON messages:
   preferred: {"q":[x,y,z,w],"t":timestamp_ms}
   fallback:  {"acc":[ax,ay,az],"gyro":[gx,gy,gz],"t":...}  (not fused here)
- Workflow:
   1) Press CONNECT and then REST CAL to collect N samples while IMU is still (computes baseline quaternion q0)
   2) Press CAPTURE and follow prompts (Right, Left, Up, Down) — perform each rotation and hold for a moment
   3) Press COMPUTE GLOBAL to produce the global axis (rotation matrix & axis vectors)
   4) Save calibration to calibration.json

Notes:
- This program averages quaternions robustly (eigen-based method).
- UI includes 3D view (pyqtgraph GL) of cube + arrows to show sensor axes in world coordinates.
"""

import sys, json, time, math, threading
from collections import deque
from dataclasses import dataclass
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import serial
import serial.tools.list_ports

# -----------------------
# Utility: quaternion math
# -----------------------
def quat_to_matrix(q):
    # q = [w,x,y,z]
    w,x,y,z = q
    R = np.array([
        [1 - 2*(y*y+z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x+z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R

def rotate_vec_by_quat(v, q):
    R = quat_to_matrix(q)
    return R.dot(v)

def normalize_quat(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])  # identity
    return q / n

def avg_quaternions(quats):
    """
    Average quaternions using the method by Markley:
    compute eigenvector of symmetric accumulator matrix.
    Input: list/array of quaternions in [w,x,y,z] form.
    """
    M = np.zeros((4,4))
    for q in quats:
        q = np.array(q, dtype=float).reshape(4,1)
        M += q @ q.T
    # Normalize by number
    M /= len(quats)
    # Compute largest eigenvector
    vals, vecs = np.linalg.eigh(M)
    q_avg = vecs[:, np.argmax(vals)]
    # Ensure normalized
    return (q_avg / np.linalg.norm(q_avg)).real

def quat_to_euler_deg(q):
    # returns (roll, pitch, yaw) in degrees using ZYX (yaw-pitch-roll)
    w,x,y,z = q
    # roll (x-axis)
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr, cosr)
    # pitch (y-axis)
    sinp = 2*(w*y - z*x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

# -----------------------
# Serial reader thread
# -----------------------
class SerialReader(QtCore.QObject):
    imu_received = QtCore.pyqtSignal(dict)  # emits parsed JSON dict per line

    def __init__(self, port, baud=115200):
        super().__init__()
        self.port_name = port
        self.baud = baud
        self._serial = None
        self._running = False
        self._thread = None

    def start(self):
        try:
            self._serial = serial.Serial(self.port_name, self.baud, timeout=1)
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {self.port_name}: {e}")
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
        except:
            pass

    def _safe_parse(self, line):
        try:
            obj = json.loads(line)
            return obj
        except Exception:
            # try to accept bare arrays like [w,x,y,z]
            try:
                arr = json.loads(line.strip())
                if isinstance(arr, list) and len(arr) >= 4:
                    return {"q": arr[:4]}
            except:
                return None
        return None

    def _read_loop(self):
        buf = b""
        while self._running:
            try:
                data = self._serial.read(1024)
                if not data:
                    time.sleep(0.005)
                    continue
                buf += data
                # split by newline
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    s = line.decode('utf8', errors='ignore').strip()
                    if not s:
                        continue
                    parsed = self._safe_parse(s)
                    if parsed is not None:
                        self.imu_received.emit(parsed)
            except Exception as e:
                # emit nothing; just sleep and continue
                print("Serial read error:", e)
                time.sleep(0.1)

# -----------------------
# Main GUI
# -----------------------
class CalibratorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Global Axis Calibrator")
        self.resize(1100, 700)
        self.setStyleSheet(self._dark_style())

        # central layout
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        hbox = QtWidgets.QHBoxLayout()
        w.setLayout(hbox)

        # left control panel
        left = QtWidgets.QFrame(); left.setMinimumWidth(380)
        left.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_layout = QtWidgets.QVBoxLayout()
        left.setLayout(left_layout)
        hbox.addWidget(left)

        # Right visual panel (3D + graphs)
        right = QtWidgets.QFrame()
        right_layout = QtWidgets.QVBoxLayout()
        right.setLayout(right_layout)
        hbox.addWidget(right, 1)

        # -------------
        # Controls
        # -------------
        title = QtWidgets.QLabel("<h2>IMU Global Axis Calibrator</h2>")
        left_layout.addWidget(title)

        # port selection
        port_h = QtWidgets.QHBoxLayout()
        self.port_combo = QtWidgets.QComboBox()
        self.refresh_ports()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        port_h.addWidget(self.port_combo)
        port_h.addWidget(self.refresh_btn)
        left_layout.addLayout(port_h)
        left_layout.addWidget(self.connect_btn)
        left_layout.addWidget(self.disconnect_btn)

        # info / live labels
        self.lbl_status = QtWidgets.QLabel("Status: Disconnected")
        self.lbl_q = QtWidgets.QLabel("Quat: -")
        self.lbl_euler = QtWidgets.QLabel("Euler (deg): -")
        self.lbl_freq = QtWidgets.QLabel("Hz: -")
        self.lbl_delta = QtWidgets.QLabel("Δq: -")
        for lbl in (self.lbl_status, self.lbl_q, self.lbl_euler, self.lbl_freq, self.lbl_delta):
            left_layout.addWidget(lbl)

        left_layout.addSpacing(8)
        # rest calibration
        self.btn_rest = QtWidgets.QPushButton("REST CALIBRATE (Hold still)")
        self.btn_rest.setToolTip("Collect N samples while IMU is held still to compute baseline.")
        left_layout.addWidget(self.btn_rest)
        self.rest_n_spin = QtWidgets.QSpinBox(); self.rest_n_spin.setRange(10,2000); self.rest_n_spin.setValue(200)
        left_layout.addWidget(QtWidgets.QLabel("Rest samples (N)"))
        left_layout.addWidget(self.rest_n_spin)

        left_layout.addSpacing(8)
        # capture gestures
        left_layout.addWidget(QtWidgets.QLabel("<b>Gesture capture</b>"))
        self.btn_capture = QtWidgets.QPushButton("CAPTURE GESTURES (Right,Left,Up,Down)")
        left_layout.addWidget(self.btn_capture)
        left_layout.addWidget(QtWidgets.QLabel("Hold each gesture when prompted. Each capture will sample for M ms."))
        self.capture_ms_spin = QtWidgets.QSpinBox(); self.capture_ms_spin.setRange(200,5000); self.capture_ms_spin.setValue(800)
        left_layout.addWidget(QtWidgets.QLabel("Capture duration (ms)"))
        left_layout.addWidget(self.capture_ms_spin)

        left_layout.addSpacing(8)
        # compute and save
        self.btn_compute = QtWidgets.QPushButton("COMPUTE GLOBAL AXIS")
        self.btn_save = QtWidgets.QPushButton("Save calibration")
        self.output_text = QtWidgets.QTextEdit(); self.output_text.setReadOnly(True); self.output_text.setFixedHeight(160)
        left_layout.addWidget(self.btn_compute)
        left_layout.addWidget(self.btn_save)
        left_layout.addWidget(QtWidgets.QLabel("Calibration output"))
        left_layout.addWidget(self.output_text)

        left_layout.addStretch(1)

        # -------------
        # Right: 3D view + graphs
        # -------------
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 5
        right_layout.addWidget(self.view, 1)

        # add a grid
        grid = gl.GLGridItem()
        grid.setSize(4,4)
        grid.setSpacing(0.5,0.5)
        self.view.addItem(grid)

        # cube representing the IMU body
        verts = np.array([
            [-0.5,-0.3,-0.8],
            [ 0.5,-0.3,-0.8],
            [ 0.5, 0.3,-0.8],
            [-0.5, 0.3,-0.8],
            [-0.5,-0.3, 0.8],
            [ 0.5,-0.3, 0.8],
            [ 0.5, 0.3, 0.8],
            [-0.5, 0.3, 0.8]
        ])
        faces = np.array([
            [0,1,2],[0,2,3],[4,5,6],[4,6,7],
            [0,1,5],[0,5,4],[2,3,7],[2,7,6],
            [1,2,6],[1,6,5],[0,3,7],[0,7,4]
        ])
        colors = np.ones((faces.shape[0],4), dtype=float) * np.array([0.45,0.7,1.0,0.9])
        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        self.cube_item = gl.GLMeshItem(meshdata=meshdata, smooth=False, drawFaces=True, drawEdges=True, edgeColor=(0.05,0.2,0.35,1.0))
        self.view.addItem(self.cube_item)

        # axis arrows
        self.axis_lines = []
        self.axis_colors = [(1,0.3,0.3,1),(0.3,1,0.4,1),(0.4,0.65,1,1)]
        for i in range(3):
            plt = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,0,0]]), width=3, color=self.axis_colors[i])
            self.view.addItem(plt)
            self.axis_lines.append(plt)

        # graphs (freq and delta)
        graph_box = QtWidgets.QHBoxLayout()
        right_layout.addLayout(graph_box)
        self.pg_freq = pg.PlotWidget(title="Sample Rate (Hz)"); self.pg_freq.setBackground((10,15,25)); self.pg_freq.showGrid(x=True,y=True,alpha=0.3)
        self.pg_delta = pg.PlotWidget(title="Δq magnitude"); self.pg_delta.setBackground((10,15,25)); self.pg_delta.showGrid(x=True,y=True,alpha=0.3)
        graph_box.addWidget(self.pg_freq)
        graph_box.addWidget(self.pg_delta)
        self.freq_curve = self.pg_freq.plot(pen=pg.mkPen('#78a1ff'))
        self.delta_curve = self.pg_delta.plot(pen=pg.mkPen('#ffd166'))
        self.freq_data = deque(maxlen=200)
        self.delta_data = deque(maxlen=200)

        # -----------------
        # Internal buffers/state
        # -----------------
        self.serial_reader = None
        self.latest_quat = None
        self.q_baseline = None
        self.rest_samples = []
        self.gesture_data = {"right": [], "left": [], "up": [], "down": []}
        self.last_timestamp = None
        self.last_time = time.time()
        self.sample_count = 0

        # Connect signals
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.handle_connect)
        self.disconnect_btn.clicked.connect(self.handle_disconnect)
        self.btn_rest.clicked.connect(self.start_rest_calibration)
        self.btn_capture.clicked.connect(self.capture_gestures)
        self.btn_compute.clicked.connect(self.compute_global_axis)
        self.btn_save.clicked.connect(self.save_calibration)

        # timer for UI updates
        self.ui_timer = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.ui_update)
        self.ui_timer.start(120)

    def _dark_style(self):
        return """
        QWidget { background: #071427; color: #e6eef6; font-family: Roboto, Arial; }
        QPushButton { background: #0f2230; border: 1px solid #12303f; padding: 6px 8px; border-radius: 6px; }
        QComboBox, QSpinBox, QLineEdit { background: #071022; color: #e6eef6; border: 1px solid #12303f; padding: 4px; border-radius:4px; }
        QTextEdit { background: #020617; color: #cfe9ff; border: 1px solid #0f2740; }
        QLabel { color: #9fb3d6; }
        """

    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.port_combo.addItem(f"{p.device} - {p.description}", p.device)

    def handle_connect(self):
        if self.serial_reader:
            self.append_output("Already connected")
            return
        port = self.port_combo.currentData()
        if port is None:
            self.append_output("No port selected")
            return
        try:
            self.serial_reader = SerialReader(port)
            self.serial_reader.imu_received.connect(self.on_imu_line)
            self.serial_reader.start()
            self.lbl_status.setText(f"Status: Connected ({port})")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.append_output(f"Connected to {port}")
            # reset counters
            self.sample_count = 0
            self.last_time = time.time()
        except Exception as e:
            self.append_output(f"Connect error: {e}")

    def handle_disconnect(self):
        if self.serial_reader:
            self.serial_reader.stop()
            self.serial_reader = None
        self.lbl_status.setText("Status: Disconnected")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.append_output("Disconnected")

    def on_imu_line(self, obj):
        # obj is parsed JSON
        # Accept q or use acc+gyro fallback (we will not fuse here)
        q = None
        if 'q' in obj and isinstance(obj['q'], (list,tuple)) and len(obj['q'])>=4:
            # assume order [x,y,z,w] or [w,x,y,z] - detect heuristics?
            arr = obj['q'][:4]
            # Heuristic: if first element near 1.0 it's probably w
            if abs(arr[0]) > 0.7:
                q = np.array([arr[0], arr[1], arr[2], arr[3]], dtype=float) # treat as [w,x,y,z]
            else:
                # assume [x,y,z,w]
                q = np.array([arr[3], arr[0], arr[1], arr[2]], dtype=float)
        elif 'acc' in obj and 'gyro' in obj:
            # fallback: compute a quick approximate quaternion from accel only for pitch/roll
            ax,ay,az = obj['acc'][:3]
            # Compute roll/pitch from accel
            pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
            roll  = math.atan2(ay, az)
            yaw = 0.0
            # convert to quaternion (ZYX)
            cy = math.cos(yaw*0.5); sy = math.sin(yaw*0.5)
            cp = math.cos(pitch*0.5); sp = math.sin(pitch*0.5)
            cr = math.cos(roll*0.5); sr = math.sin(roll*0.5)
            w = cr*cp*cy + sr*sp*sy
            x = sr*cp*cy - cr*sp*sy
            y = cr*sp*cy + sr*cp*sy
            z = cr*cp*sy - sr*sp*cy
            q = np.array([w,x,y,z], dtype=float)
        else:
            # ignore unknown
            return

        q = normalize_quat(q)
        self.latest_quat = q
        self.sample_count += 1

        # compute delta magnitude relative to previous
        if hasattr(self, "_last_q"):
            dq = q - self._last_q
            dmag = np.linalg.norm(dq)
        else:
            dmag = 0.0
        self._last_q = q
        self.delta_data.append(dmag)

        now = time.time()
        if now - self.last_time >= 1.0:
            hz = self.sample_count / (now - self.last_time)
            self.freq_data.append(hz)
            self.sample_count = 0
            self.last_time = now

    def append_output(self, s):
        ts = time.strftime("%H:%M:%S")
        self.output_text.append(f"[{ts}] {s}")

    def start_rest_calibration(self):
        if self.latest_quat is None:
            self.append_output("No data yet to calibrate from.")
            return
        N = int(self.rest_n_spin.value())
        self.append_output(f"Starting rest calibration: collecting {N} samples. Hold IMU very still.")
        # blocking collection in a thread so UI remains responsive
        def worker():
            samples = []
            start_t = time.time()
            timeout = 10.0
            while len(samples) < N and (time.time() - start_t) < timeout:
                if self.latest_quat is not None:
                    samples.append(self.latest_quat.copy())
                time.sleep(1.0/200.0)
            if len(samples) == 0:
                self.append_output("No samples collected.")
                return
            qavg = avg_quaternions(samples)
            self.q_baseline = qavg
            self.append_output("Rest baseline quaternion set.")
            e = quat_to_euler_deg(qavg)
            self.append_output(f"Baseline Euler (deg): roll={e[0]:.2f} pitch={e[1]:.2f} yaw={e[2]:.2f}")
        threading.Thread(target=worker, daemon=True).start()

    def capture_gestures(self):
        if self.latest_quat is None:
            self.append_output("No IMU data present. Connect first.")
            return
        # sequence: right, left, up, down
        seq = [("right","Rotate right and hold"), ("left","Rotate left and hold"),
               ("up","Rotate up (pitch up) and hold"), ("down","Rotate down (pitch down) and hold")]
        duration_ms = int(self.capture_ms_spin.value())
        self.append_output(f"Starting gesture capture: duration {duration_ms} ms per gesture.")
        def worker():
            for key, prompt in seq:
                self.append_output(f"Prepare: {prompt} ... starting in 1.5s")
                time.sleep(1.5)
                self.append_output(f"Capturing {key} for {duration_ms} ms ...")
                samples = []
                start = time.time()
                while (time.time() - start) < (duration_ms/1000.0):
                    if self.latest_quat is not None:
                        samples.append(self.latest_quat.copy())
                    time.sleep(1.0/200.0)
                if len(samples) == 0:
                    self.append_output(f"No samples for {key}")
                else:
                    qavg = avg_quaternions(samples)
                    self.gesture_data[key] = samples  # store raw list
                    self.append_output(f"{key} captured: {len(samples)} samples")
            self.append_output("Gesture capture finished.")
        threading.Thread(target=worker, daemon=True).start()

    def compute_global_axis(self):
        # Need baseline and gesture captures else abort
        if self.q_baseline is None:
            self.append_output("Baseline not set. Run REST CALIBRATE first.")
            return
        # For safety, ensure we have captures. If some missing, we can still compute axes from baseline.
        # Strategy:
        #  - Use baseline quaternion q0 as sensor->world reference.
        #  - Transform sensor basis vectors to world: columns of R0 = R(q0) * e_i
        q0 = self.q_baseline
        R0 = quat_to_matrix(q0)  # rotation sensor->world
        # sensor axes in world:
        sx = R0[:,0]
        sy = R0[:,1]
        sz = R0[:,2]
        # If gestures available, compute mean relative rotations and give more robust axes:
        def mean_gesture_quat(key):
            arr = self.gesture_data.get(key, [])
            if not arr: return None
            return avg_quaternions(arr)
        right_q = mean_gesture_quat("right")
        left_q  = mean_gesture_quat("left")
        up_q    = mean_gesture_quat("up")
        down_q  = mean_gesture_quat("down")

        # Build a result dict for output & save
        res = {}
        res['baseline_quaternion'] = q0.tolist()
        res['baseline_euler_deg'] = dict(zip(["roll","pitch","yaw"], quat_to_euler_deg(q0)))
        res['R_sensor_to_world'] = R0.tolist()
        res['sensor_axes_world'] = {'x': sx.tolist(), 'y': sy.tolist(), 'z': sz.tolist()}

        # If we have right/left pair, compute average yaw axis vector, etc.
        # Here we compute the direction of motion for right-left as axis in world.
        def rel_axis(q_capture):
            # axis of rotation approximated from quaternion relative to baseline:
            # q_rel = inv(q0) * q_capture
            if q_capture is None: return None
            q_rel = quat_mul(quat_inv(q0), q_capture)
            # extract axis-angle: axis = (x,y,z)/sin(theta/2)
            w,x,y,z = q_rel
            sin_half = math.sqrt(max(0.0, 1 - w*w))
            if sin_half < 1e-6:
                return None
            axis = np.array([x/sin_half, y/sin_half, z/sin_half], dtype=float)
            axis = axis / np.linalg.norm(axis)
            return axis

        # helper quaternion functions inline
        def quat_inv(q):
            q = np.array(q, dtype=float)
            return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
        def quat_mul(a,b):
            # a,b as [w,x,y,z]
            w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dtype=float)

        # compute relative axes
        r_axis = rel_axis(right_q)
        l_axis = rel_axis(left_q)
        u_axis = rel_axis(up_q)
        d_axis = rel_axis(down_q)

        res['rel_axes'] = {
            'right': r_axis.tolist() if r_axis is not None else None,
            'left' : l_axis.tolist() if l_axis is not None else None,
            'up'   : u_axis.tolist() if u_axis is not None else None,
            'down' : d_axis.tolist() if d_axis is not None else None
        }

        # Determine a "global up" estimate: prefer up/down if available, else use baseline z-axis
        global_up = None
        if u_axis is not None and d_axis is not None:
            global_up = (u_axis - d_axis)
            if np.linalg.norm(global_up) > 1e-6:
                global_up = global_up / np.linalg.norm(global_up)
        elif u_axis is not None:
            global_up = u_axis
        elif d_axis is not None:
            global_up = d_axis
        else:
            global_up = sz  # fallback to baseline sensor z

        # Determine global forward (right-left) axis: prefer right-left pair
        global_right = None
        if r_axis is not None and l_axis is not None:
            global_right = (r_axis - l_axis)
            if np.linalg.norm(global_right) > 1e-6:
                global_right = global_right / np.linalg.norm(global_right)
        elif r_axis is not None:
            global_right = r_axis
        elif l_axis is not None:
            global_right = l_axis
        else:
            global_right = sx

        # Make orthonormal triad: right, up, forward
        # ensure orthogonality: recompute forward = cross(up, right)
        up = np.array(global_up, dtype=float)
        right_v = np.array(global_right, dtype=float)
        # orthogonalize right relative to up
        right_v = right_v - up * np.dot(up, right_v)
        if np.linalg.norm(right_v) < 1e-6:
            right_v = np.array([1.0,0.0,0.0])
        right_v = right_v / np.linalg.norm(right_v)
        forward_v = np.cross(up, right_v)
        forward_v = forward_v / np.linalg.norm(forward_v)

        res['global_axes_estimate'] = {'up': up.tolist(), 'right': right_v.tolist(), 'forward': forward_v.tolist()}

        # show results in UI and set internal global axis - we build a rotation matrix from estimated axes
        R_global = np.column_stack([right_v, up, forward_v])  # columns are new axes
        res['R_global'] = R_global.tolist()

        # Display nicely
        out = json.dumps(res, indent=2)
        self.output_text.setPlainText(out)
        # Update arrows/cube to show sensor axes in world (sensor basis transformed by baseline)
        self.update_visual_axes(R0, right_v, up, forward_v)

        # Save last result in object
        self.last_result = res
        self.append_output("Computed global axis. Review output and Save if OK.")

    def update_visual_axes(self, R0, right_v, up_v, forward_v):
        # Update axis lines representing sensor basis transformed by baseline
        # We'll plot sensor axes (R0 columns) scaled
        sx = R0[:,0]; sy = R0[:,1]; sz = R0[:,2]
        scale = 1.5
        axes = [sx, sy, sz]
        for i, arr in enumerate(axes):
            pts = np.array([[0,0,0],[arr[0]*scale, arr[1]*scale, arr[2]*scale]])
            self.axis_lines[i].setData(pos=pts)

    def save_calibration(self):
        if not hasattr(self, "last_result") or self.last_result is None:
            self.append_output("No calibration result to save.")
            return
        fname = "calibration.json"
        try:
            with open(fname, "w") as f:
                json.dump(self.last_result, f, indent=2)
            self.append_output(f"Saved calibration to {fname}")
        except Exception as e:
            self.append_output(f"Save error: {e}")

    def ui_update(self):
        # update live labels + graphs
        if self.latest_quat is not None:
            q = self.latest_quat
            self.lbl_q.setText(f"Quat: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
            e = quat_to_euler_deg(q)
            self.lbl_euler.setText(f"Euler (deg): roll={e[0]:.2f}, pitch={e[1]:.2f}, yaw={e[2]:.2f}")
        if len(self.freq_data) > 0:
            arr = list(self.freq_data)
            self.freq_curve.setData(arr)
            self.lbl_freq.setText(f"Hz: {arr[-1]:.1f}")
        if len(self.delta_data) > 0:
            darr = list(self.delta_data)
            self.delta_curve.setData(darr)
            self.lbl_delta.setText(f"Δq: {darr[-1]:.6f}")
        # also show baseline if present
        if self.q_baseline is not None:
            e = quat_to_euler_deg(self.q_baseline)
            self.lbl_status.setText(f"Baseline set. Euler r={e[0]:.2f}, p={e[1]:.2f}, y={e[2]:.2f}")
        # update virtual cube orientation to latest (apply baseline inverse so we show relative)
        if self.latest_quat is not None:
            q = self.latest_quat
            # show cube at q relative to baseline if baseline exists
            if self.q_baseline is not None:
                q_rel = quat_mul(quat_inv(self.q_baseline), q)
                q_disp = q_rel
            else:
                q_disp = q
            # convert to rotation matrix and apply to cube mesh transform
            R = quat_to_matrix(q_disp)
            # pyqtgraph GLMeshItem has setTransform; but easiest is to create quaternion -> rotation matrix -> 4x4 transform
            M = np.eye(4)
            M[:3,:3] = R
            # pyqtgraph expects column-major flatten
            transform = QtGui.QMatrix4x4(*M.T.flatten())
            try:
                self.cube_item.setTransform(transform)
            except Exception:
                pass

# quaternion helpers for internal use
def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
def quat_mul(a,b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

# -----------------------
# Run app
# -----------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = CalibratorApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
