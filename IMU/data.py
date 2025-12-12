# udp_serial_bridge.py
import socket
import threading
import serial
import time

BRIDGE_SERIAL = "COM16"   # bridge writes to this; other app listens on COM15
SER_BAUD = 115200
UDP_LISTEN_IP = "0.0.0.0"
UDP_LISTEN_PORT = 4212    # must match ESP udpPort
SER_READ_TIMEOUT = 0.2

# If you want serial data NOT to be forwarded back to UDP, set to False
FORWARD_SERIAL_TO_UDP = True

# Globals used to remember last UDP sender so serial->UDP can reply
last_udp_addr = None
last_udp_lock = threading.Lock()

def udp_listener_thread(ser):
    """
    Listen for UDP packets and write payloads to serial (as bytes).
    Remember last sender address for optional replies.
    """
    global last_udp_addr
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_LISTEN_IP, UDP_LISTEN_PORT))
    sock.setblocking(True)
    print(f"[UDP] Listening on {UDP_LISTEN_IP}:{UDP_LISTEN_PORT}")
    while True:
        try:
            data, addr = sock.recvfrom(8192)
            if not data:
                continue
            # remember sender
            with last_udp_lock:
                last_udp_addr = addr
            # ensure newline termination so serial consumers get lines
            if not data.endswith(b'\n'):
                data = data + b'\n'
            # write to serial
            try:
                ser.write(data)
                ser.flush()
            except Exception as e:
                print("[UDP->SER] Serial write error:", e)
        except Exception as e:
            print("[UDP] Listener error:", e)
            time.sleep(0.5)

def serial_reader_thread(ser):
    """
    Read from serial and optionally forward to last UDP sender.
    """
    global last_udp_addr
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        try:
            data = ser.readline()  # reads up to newline
            if not data:
                time.sleep(0.005)
                continue
            # strip CR/LF
            payload = data.rstrip(b'\r\n')
            print(f"[SER->] {payload!r}")
            if FORWARD_SERIAL_TO_UDP:
                with last_udp_lock:
                    addr = last_udp_addr
                if addr:
                    try:
                        sock.sendto(payload + b'\n', addr)
                    except Exception as e:
                        print("[SER->UDP] send error:", e)
        except Exception as e:
            print("[SER] read error:", e)
            time.sleep(0.5)

def main():
    # open serial
    while True:
        try:
            ser = serial.Serial(BRIDGE_SERIAL, SER_BAUD, timeout=SER_READ_TIMEOUT)
            print(f"[SER] Opened {BRIDGE_SERIAL} @ {SER_BAUD}")
            break
        except Exception as e:
            print(f"[SER] Could not open {BRIDGE_SERIAL}: {e}. Retrying in 2s...")
            time.sleep(2)

    t_udp = threading.Thread(target=udp_listener_thread, args=(ser,), daemon=True)
    t_ser = threading.Thread(target=serial_reader_thread, args=(ser,), daemon=True)
    t_udp.start()
    t_ser.start()

    print("Bridge running. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping bridge.")
        try:
            ser.close()
        except:
            pass

if __name__ == "__main__":
    main()
