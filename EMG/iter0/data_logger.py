import serial
import time
import csv
import sys

# ==========================================
# CONFIGURATION
# ==========================================
COM_PORT = 'COM5'       # <--- CHANGE THIS to your Arduino Port
BAUD_RATE = 115200      # Must match your Arduino code
FILE_NAME = "gesture_FIST_01.csv" # <--- CHANGE THIS for every new recording!
# ==========================================

def collect_data():
    # 1. Setup Serial Connection
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"✅ Connected to {COM_PORT}")
        time.sleep(2) # Wait for Arduino to reset
    except serial.SerialException:
        print(f"❌ Could not open {COM_PORT}. Is the Arduino IDE Serial Monitor still open?")
        sys.exit()

    data_buffer = []
    print(f"\n🔴 RECORDING STARTED: {FILE_NAME}")
    print("Perform your gesture repeatedly...")
    print("Press 'Ctrl + C' to stop and save.\n")

    # 2. Main Loop
    try:
        while True:
            if ser.in_waiting > 0:
                # Read line from Arduino, decode bytes to string, strip whitespace
                raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Ensure it's a valid number
                if raw_line.isdigit():
                    val = int(raw_line)
                    data_buffer.append([val])
                    
                    # Visual feedback (prints a dot every 50 samples)
                    if len(data_buffer) % 50 == 0:
                        print(".", end="", flush=True)

    # 3. Save on Exit
    except KeyboardInterrupt:
        print(f"\n\n🛑 Recording Stopped.")
        print(f"💾 Saving {len(data_buffer)} samples to '{FILE_NAME}'...")
        
        with open(FILE_NAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["EMG_Value"]) # Header column
            writer.writerows(data_buffer)
        
        print("✅ Data saved successfully!")
        ser.close()

if __name__ == "__main__":
    collect_data()