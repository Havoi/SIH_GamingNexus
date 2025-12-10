import serial
import time
import pyautogui
import math

# CONFIG
PORT = 'COM10'  # Check this!
BAUD = 115200

print("--- DIAGNOSTIC MODE ---")
print(f"1. Testing Mouse Permissions...")
try:
    # Try to wiggle the mouse 10 pixels
    current_x, current_y = pyautogui.position()
    pyautogui.moveRel(10, 0)
    pyautogui.moveRel(-10, 0)
    print("   [PASS] Mouse moved successfully.")
except Exception as e:
    print(f"   [FAIL] Windows blocked the mouse! Run as Administrator. Error: {e}")

print(f"2. Connecting to {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    # FORCE ESP32 RESTART (DTR Toggle)
    ser.dtr = False
    time.sleep(0.1)
    ser.dtr = True
    print("   [PASS] Serial Port Opened. Listening for data...")
except Exception as e:
    print(f"   [FAIL] Could not open port! Close browser/other apps. Error: {e}")
    exit()

print("\n3. READING STREAM (Press Ctrl+C to stop)")
print("   If you see lines appearing below, the hardware is fine.")
print("   Format: [Raw String] -> [Parsed Quat]")

while True:
    if ser.in_waiting:
        try:
            # Read line
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # Filter for IMU data
            if '"type":"imu"' in line:
                print(f"   DATA OK: {line[:60]}...") # Print first 60 chars
            else:
                # Print garbage/debug data from ESP32
                print(f"   RAW: {line}")
                
        except Exception as e:
            print(f"   [ERROR] Parse fail: {e}")
    else:
        # If nothing happens for a while
        pass