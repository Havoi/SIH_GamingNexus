"""
data_sources.py

Data source abstractions for EMG + IMU streaming.

Provides two threaded classes:
- SerialSource: reads from a real serial device
- SimulatedSource: generates synthetic EMG + IMU data for development

Each class pushes data into a queue as tuples:
    ('emg_sample', float_value)
    ('imu_sample', {'acc': np.array([ax,ay,az]), 'gyro': np.array([gx,gy,gz])})
    ('info', 'message')
    ('error', 'message')

The GUI or controller reads from this queue and processes samples.
"""
from __future__ import annotations
import threading
import time
import serial
import numpy as np
import random


# ---------------------------------------------
# Serial data source
# ---------------------------------------------
class SerialSource(threading.Thread):
    def __init__(self, port: str, baud: int, out_queue, timeout: float = 1.0):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.out_queue = out_queue
        self._stop_event = threading.Event()

        self.ser = None

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        except Exception as e:
            self.out_queue.put(("error", f"Failed to open serial {self.port}@{self.baud}: {e}"))
            return

        self.out_queue.put(("info", f"Serial connected {self.port}@{self.baud}"))

        try:
            while not self.stopped():
                line = ''
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                except Exception:
                    continue

                if not line:
                    time.sleep(0.001)
                    continue

                parts = [p.strip() for p in line.split(',') if p.strip() != '']
                try:
                    vals = [float(p) for p in parts]
                except Exception:
                    continue

                # EMG: single value
                if len(vals) == 1:
                    self.out_queue.put(('emg_sample', float(vals[0])))

                # IMU: 3 or 6 values → acc + optional gyro
                elif len(vals) >= 3:
                    if len(vals) >= 6:
                        acc = np.array(vals[:3], dtype=float)
                        gyro = np.array(vals[3:6], dtype=float)
                    else:
                        acc = np.array(vals[:3], dtype=float)
                        gyro = np.zeros(3, dtype=float)
                    self.out_queue.put(('imu_sample', {'acc': acc, 'gyro': gyro}))

        except Exception as e:
            self.out_queue.put(("error", f"Serial thread error: {e}"))

        finally:
            try:
                if self.ser:
                    self.ser.close()
            except Exception:
                pass
            self.out_queue.put(("info", "Serial closed"))


# ---------------------------------------------
# Simulated data source (dev mode)
# ---------------------------------------------
class SimulatedSource(threading.Thread):
    """Generates synthetic EMG + IMU data at realistic sampling rates.

    EMG → 100 Hz with random bursts
    IMU → 50 Hz with noisy acc/gyro
    """
    def __init__(self, out_queue, emg_rate_hz=100, imu_rate_hz=50):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.emg_rate = emg_rate_hz
        self.imu_rate = imu_rate_hz
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        t = 0.0
        dt = 0.001
        next_emg = 0.0
        next_imu = 0.0
        burst_interval = 3.0

        while not self.stopped():
            # EMG generation
            if t >= next_emg:
                burst = (1.0 if (int(t) % int(burst_interval) == 0) else 0.0)
                noise = 0.02 * random.uniform(-1, 1)
                val = noise + burst * (0.6 * random.uniform(0.8, 1.2))
                self.out_queue.put(('emg_sample', float(val)))
                next_emg += 1.0 / self.emg_rate

            # IMU generation
            if t >= next_imu:
                acc = np.array([0.0, 0.0, 9.81]) + 0.2 * np.random.randn(3)
                gyro = 0.01 * np.random.randn(3)
                self.out_queue.put(('imu_sample', {'acc': acc, 'gyro': gyro}))
                next_imu += 1.0 / self.imu_rate

            time.sleep(dt)
            t += dt
