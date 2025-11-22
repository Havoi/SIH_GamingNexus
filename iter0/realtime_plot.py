import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading

# ==========================================
# CONFIGURATION
# ==========================================
COM_PORT = 'COM5'       # <--- Your Port
BAUD_RATE = 115200      # Must match Arduino
MAX_SAMPLES = 500       # How much history to show on screen
Y_MIN = 0               # Min value (usually 0)
Y_MAX = 1024            # Max value (Arduino ADC limit)
# ==========================================

# Initialize Serial Connection
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"✅ Connected to {COM_PORT}. Waiting for data...")
    time.sleep(2) # Allow Arduino to reset
except serial.SerialException:
    print(f"❌ ERROR: Could not open {COM_PORT}.")
    print("👉 Solution: Close Arduino IDE Serial Monitor or any other Python script.")
    exit()

# Data Buffer (Efficient Queue)
data_buffer = deque([0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)

# Setup the Plot
fig, ax = plt.subplots()
line, = ax.plot(data_buffer)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_title(f"Real-Time EMG Signal ({COM_PORT})")
ax.set_ylabel("Signal Amplitude (0-1024)")
ax.set_xlabel("Time")
ax.grid(True)

# This function runs in the background to read Serial data continuously
# We use a thread so the GUI doesn't freeze while waiting for data
def read_serial():
    while True:
        try:
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                if raw_line.isdigit():
                    val = int(raw_line)
                    data_buffer.append(val)
        except:
            pass

# Start the background reader thread
thread = threading.Thread(target=read_serial, daemon=True)
thread.start()

# This function updates the graph
def update_plot(frame):
    line.set_ydata(data_buffer)
    return line,

# Start Animation
print("📈 Plotting... Close the graph window to stop.")
ani = animation.FuncAnimation(fig, update_plot, interval=30, blit=True)
plt.show()

# Clean exit
ser.close()
print("Disconnected.")