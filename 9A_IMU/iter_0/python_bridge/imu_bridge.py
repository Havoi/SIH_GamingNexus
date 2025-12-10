# imu_bridge_serial.py
# Python bridge that reads IMU JSON lines from serial (USB),
# provides calibration and mouse emulation, and hosts a WebSocket for dashboard.

import asyncio, websockets, json, sys, argparse, time, math
import numpy as np
from collections import deque
import pyautogui
import serial

# ---------- default config ----------
DASH_WS_PORT = 8766
BAUD = 115200
BASE_MOUSE_SCALE = 1200.0  # pixels per radian base
sensitivity = 1.0
smoothing_alpha = 0.2
deadzone = 0.02  # radians

pyautogui.FAILSAFE = False

# ---------- runtime state ----------
baseline_q = None
last_quat = None
smoothed_dx = 0.0
smoothed_dy = 0.0
dash_clients = set()

# ---------- quaternion helpers ----------
def quat_mul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]
def quat_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]
def quat_inv(q):
    # assume unit quaternion
    return quat_conj(q)
def quat_to_euler(q):
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
    return roll, pitch, yaw

# ---------- broadcast to dashboards ----------
async def broadcast(obj):
    js = json.dumps(obj)
    for c in list(dash_clients):
        try:
            await c.send(js)
        except:
            pass

async def handle_dashboard(ws, path):
    print("[DASH] connected")
    dash_clients.add(ws)
    try:
        async for msg in ws:
            try:
                j = json.loads(msg)
            except:
                continue
            # process commands from dashboard
            if j.get('cmd') == 'calibrate':
                # set baseline to last_quat if available
                global baseline_q
                if last_quat is not None:
                    baseline_q = last_quat.copy()
                    print("[CAL] baseline set from dashboard")
                    await broadcast({"type":"status","msg":"baseline_set"})
            elif j.get('cmd') == 'set':
                k = j.get('k'); v = j.get('v')
                if k == 'sensitivity':
                    global sensitivity
                    sensitivity = float(v)
                elif k == 'smoothing':
                    global smoothing_alpha
                    smoothing_alpha = float(v)
                elif k == 'deadzone':
                    global deadzone
                    deadzone = float(v)
                print(f"[DASH] set {k}={v}")
                await broadcast({"type":"status","msg":f"set {k}={v}"})
    finally:
        dash_clients.remove(ws)
        print("[DASH] disconnected")

# ---------- serial reader (blocking in thread) ----------
def start_serial_reader(port, baud, loop):
    ser = serial.Serial(port, baud, timeout=1)
    print(f"[SERIAL] opened {port} @ {baud}")
    # run a small thread that pushes lines into asyncio loop
    def reader():
        nonlocal ser, loop
        while True:
            try:
                line = ser.readline().decode('utf8', errors='ignore').strip()
                if not line:
                    continue
                # feed into asyncio by scheduling a coroutine
                asyncio.run_coroutine_threadsafe(process_serial_line(line), loop)
            except Exception as e:
                print("Serial read error:", e)
                time.sleep(0.2)
    import threading
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    return ser

# ---------- process IMU JSON line ----------
async def process_serial_line(line):
    global last_quat, baseline_q, smoothed_dx, smoothed_dy
    # try parse JSON
    try:
        j = json.loads(line)
    except Exception as e:
        print("[SERIAL] non-json:", line[:120])
        return
    # handle special types
    if j.get('type') == 'imu' and 'quat' in j:
        quat = j['quat']  # [w,x,y,z]
        last_quat = quat
        # broadcast raw imu to dashboards
        await broadcast({**j, "source":"esp"})
        # if baseline not set, keep first frame auto (optional)
        if baseline_q is None:
            baseline_q = quat.copy()
            print("[CAL] baseline auto-set")
        # relative quaternion
        q_inv = quat_inv(baseline_q)
        q_rel = quat_mul(q_inv, quat)
        roll, pitch, yaw = quat_to_euler(q_rel)  # radians
        # deadzone
        if abs(yaw) < deadzone: yaw = 0.0
        if abs(pitch) < deadzone: pitch = 0.0
        # map yaw -> dx, pitch -> dy (negate as needed)
        dx = sensitivity * BASE_MOUSE_SCALE * yaw
        dy = -sensitivity * BASE_MOUSE_SCALE * pitch
        # smoothing EMA
        smoothed_dx = smoothing_alpha * dx + (1.0 - smoothing_alpha) * smoothed_dx
        smoothed_dy = smoothing_alpha * dy + (1.0 - smoothing_alpha) * smoothed_dy
        # move mouse relative
        try:
            sx = int(max(-3000, min(3000, smoothed_dx)))
            sy = int(max(-3000, min(3000, smoothed_dy)))
            if sx != 0 or sy != 0:
                pyautogui.moveRel(sx, sy)
        except Exception as e:
            print("Mouse move error:", e)
    elif j.get('type') == 'cal_request':
        # ESP button pressed - set baseline to last_quat
        if last_quat is not None:
            baseline_q = last_quat.copy()
            print("[CAL] baseline set from esp button")
            await broadcast({"type":"status","msg":"baseline_set_from_esp"})
    elif j.get('type') == 'param_ack' or j.get('type') == 'ack':
        # forward to dashboard
        await broadcast(j)
    else:
        # forward everything else to dashboard
        await broadcast(j)

# ---------- main ----------
async def main_loop(args):
    loop = asyncio.get_running_loop()
    # start serial reader thread
    ser = start_serial_reader(args.port, args.baud, loop)
    # launch websocket server for dashboard
    dash_server = await websockets.serve(handle_dashboard, "0.0.0.0", DASH_WS_PORT)
    print(f"[WS] dashboard server listening ws://0.0.0.0:{DASH_WS_PORT}")
    # keep alive
    await asyncio.Future()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port (eg COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=BAUD)
    args = parser.parse_args()
    try:
        asyncio.run(main_loop(args))
    except KeyboardInterrupt:
        print("Exiting")
