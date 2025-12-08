/* ------------------------------------------------------------------
   Nano R4 Minima + MPU6050 (no magnetometer) - Complete Project Sketch
   - Robust I2C reads (Wire.begin(), fixed A4/A5 pins)
   - Madgwick IMU-only filter (inline implementation)
   - JSON quaternion streaming: {"q":[x,y,z,w],"t":<micros>}
   - Serial commands: CAL_GYRO, SET_FREQ:NN, SET_BETA:val, INFO, SAVE, LOAD, RESET_BIAS
   - Saves gyro biases to EEPROM
   - Author: adapted for your setup (no external Madgwick library needed)
   ------------------------------------------------------------------*/

#include <Wire.h>
#include <EEPROM.h> // standard Arduino EEPROM (works on Nano R4 via Arduino core)

#define SERIAL_BAUD 115200

// MPU6050 registers & address
const uint8_t MPU_ADDR = 0x68;         // AD0 tied to GND
const uint8_t WHO_AM_I_REG = 0x75;
const uint8_t PWR_MGMT_1 = 0x6B;
const uint8_t ACCEL_XOUT_H = 0x3B;

// Sampling defaults
float SAMPLE_FREQ = 100.0f;            // Hz default
unsigned long sample_interval_us = (unsigned long)(1e6f / SAMPLE_FREQ);
unsigned long last_sample_time_us = 0;

// Madgwick filter state (q0 = w, q1..3 = x,y,z)
float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
float beta = 0.1f;                     // Madgwick gain, tweakable

// Gyro bias (deg/s)
float gyroBiasX = 0.0f, gyroBiasY = 0.0f, gyroBiasZ = 0.0f;

// Command parser
String cmdBuf = "";

// EEPROM storage addresses (simple layout)
const int EEPROM_FLAG_ADDR = 0;         // byte: 0xA5 if biases saved
const int EEPROM_BIAS_ADDR = 4;         // float x3 => 12 bytes (addresses 4..15)

// Safety & status
unsigned long lastStatusMs = 0;
const unsigned long STATUS_INTERVAL_MS = 10000;

// Forward declarations
void MadgwickUpdateIMU(float gx, float gy, float gz, float ax, float ay, float az, float dt);
bool readRegisters(uint8_t dev, uint8_t startReg, uint8_t *buffer, uint8_t len, unsigned int timeoutMs = 200);
bool writeRegister(uint8_t dev, uint8_t reg, uint8_t val);
void calibrateGyro(int samples = 500);
void saveBiasesToEEPROM();
bool loadBiasesFromEEPROM();
void processSerialCommands();
void printInfo();
void sendQuaternionJSON(unsigned long tmicros);
void resetBiases();

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) ; // wait for Serial
  Serial.println("\n=== Nano R4 + MPU6050 (IMU-only) - Madgwick AHRS ===");

  Wire.begin();                // Nano R4 uses fixed SDA=A4, SCL=A5
  Wire.setClock(100000);       // safe default

  // Probe WHO_AM_I
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(WHO_AM_I_REG);
  if (Wire.endTransmission(false) == 0) {
    Wire.requestFrom((int)MPU_ADDR, 1);
    if (Wire.available()) {
      uint8_t who = Wire.read();
      Serial.print("WHO_AM_I = 0x"); Serial.println(who, HEX);
    }
  } else {
    Serial.println("WARNING: MPU not responding to WHO_AM_I probe.");
  }

  // Wake device
  if (!writeRegister(MPU_ADDR, PWR_MGMT_1, 0x00)) {
    Serial.println("WARNING: could not wake MPU (I2C err)");
  } else {
    Serial.println("MPU wake OK.");
  }

  // load biases if saved
  if (loadBiasesFromEEPROM()) {
    Serial.print("Loaded biases from EEPROM: ");
    Serial.print(gyroBiasX,4); Serial.print(", "); Serial.print(gyroBiasY,4); Serial.print(", "); Serial.println(gyroBiasZ,4);
  } else {
    Serial.println("No saved biases found in EEPROM.");
  }

  sample_interval_us = (unsigned long)(1e6f / SAMPLE_FREQ);
  last_sample_time_us = micros();
  lastStatusMs = millis();

  Serial.println("Ready. Commands: CAL_GYRO, SET_FREQ:NN, SET_BETA:0.0..1.0, INFO, SAVE, LOAD, RESET_BIAS");
}

// ---------------- main loop ----------------
void loop() {
  processSerialCommands();

  unsigned long now = micros();
  if ((now - last_sample_time_us) < sample_interval_us) return;
  unsigned long t0 = now;
  // maintain consistent sampling time
  last_sample_time_us += sample_interval_us;

  uint8_t buf[14];
  if (!readRegisters(MPU_ADDR, ACCEL_XOUT_H, buf, 14, 200)) {
    Serial.println("[ERR] I2C read failed");
    return;
  }

  int16_t ax_raw = (buf[0] << 8) | buf[1];
  int16_t ay_raw = (buf[2] << 8) | buf[3];
  int16_t az_raw = (buf[4] << 8) | buf[5];
  int16_t temp_raw = (buf[6] << 8) | buf[7];
  int16_t gx_raw = (buf[8] << 8) | buf[9];
  int16_t gy_raw = (buf[10] << 8) | buf[11];
  int16_t gz_raw = (buf[12] << 8) | buf[13];

  // Convert to physical units (assume default ranges: accel ±2g, gyro ±250 dps)
  float ax = ax_raw / 16384.0f;
  float ay = ay_raw / 16384.0f;
  float az = az_raw / 16384.0f;

  float gx = gx_raw / 131.0f; // deg/s
  float gy = gy_raw / 131.0f;
  float gz = gz_raw / 131.0f;

  // remove gyro bias and convert to rad/s
  const float d2r = 3.14159265358979323846f / 180.0f;
  float gx_r = (gx - gyroBiasX) * d2r;
  float gy_r = (gy - gyroBiasY) * d2r;
  float gz_r = (gz - gyroBiasZ) * d2r;

  float dt = (sample_interval_us) * 1e-6f;

  // Update Madgwick filter
  MadgwickUpdateIMU(gx_r, gy_r, gz_r, ax, ay, az, dt);

  // Output quaternion JSON
  sendQuaternionJSON(t0);

  // Periodic status
  if (millis() - lastStatusMs > STATUS_INTERVAL_MS) {
    lastStatusMs = millis();
    Serial.print("[STATUS] freq="); Serial.print(SAMPLE_FREQ); Serial.print("Hz beta=");
    Serial.print(beta, 3); Serial.print(" biases(deg/s)=");
    Serial.print(gyroBiasX,4); Serial.print(","); Serial.print(gyroBiasY,4); Serial.print(","); Serial.println(gyroBiasZ,4);
  }
}

// ---------------- serial cmd handling ----------------
void processSerialCommands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (cmdBuf.length() > 0) {
        String cmd = cmdBuf;
        cmdBuf = "";
        cmd.trim();
        cmd.toUpperCase();
        if (cmd == "CAL_GYRO") {
          calibrateGyro(600);
        } else if (cmd.startsWith("SET_FREQ:")) {
          int v = cmd.substring(9).toInt();
          if (v >= 10 && v <= 400) {
            SAMPLE_FREQ = (float)v;
            sample_interval_us = (unsigned long)(1e6f / SAMPLE_FREQ);
            Serial.print("[CMD] SAMPLE_FREQ set to "); Serial.println(SAMPLE_FREQ);
          } else Serial.println("[CMD] invalid freq (10..400)");
        } else if (cmd.startsWith("SET_BETA:")) {
          float b = cmd.substring(9).toFloat();
          if (b >= 0.0f && b <= 1.0f) {
            beta = b;
            Serial.print("[CMD] beta set to "); Serial.println(beta, 6);
          } else Serial.println("[CMD] invalid beta (0..1)");
        } else if (cmd == "INFO") {
          printInfo();
        } else if (cmd == "SAVE") {
          saveBiasesToEEPROM();
        } else if (cmd == "LOAD") {
          if (loadBiasesFromEEPROM()) {
            Serial.println("[CMD] biases loaded.");
          } else Serial.println("[CMD] no biases saved.");
        } else if (cmd == "RESET_BIAS") {
          resetBiases();
          Serial.println("[CMD] biases reset to zero.");
        } else {
          Serial.print("[CMD] Unknown: "); Serial.println(cmd);
        }
      }
    } else {
      cmdBuf += c;
      if (cmdBuf.length() > 120) cmdBuf = cmdBuf.substring(cmdBuf.length()-120); // keep tail
    }
  }
}

void printInfo() {
  Serial.print("INFO: freq="); Serial.print(SAMPLE_FREQ); Serial.print("Hz beta=");
  Serial.print(beta, 3); Serial.print(" biases(deg/s)=");
  Serial.print(gyroBiasX,4); Serial.print(","); Serial.print(gyroBiasY,4); Serial.print(","); Serial.println(gyroBiasZ,4);
}

// ---------------- calibration ----------------
void calibrateGyro(int samples) {
  Serial.println("[CAL] Keep IMU perfectly still. Gathering samples...");
  double sx = 0, sy = 0, sz = 0;
  int valid = 0;
  uint8_t buf[14];
  for (int i = 0; i < samples; ++i) {
    if (!readRegisters(MPU_ADDR, ACCEL_XOUT_H, buf, 14, 500)) {
      Serial.println("[CAL] I2C read failed during calibration!");
      delay(50);
      continue;
    }
    int16_t gx = (buf[8] << 8) | buf[9];
    int16_t gy = (buf[10] << 8) | buf[11];
    int16_t gz = (buf[12] << 8) | buf[13];
    float gx_dps = gx / 131.0f;
    float gy_dps = gy / 131.0f;
    float gz_dps = gz / 131.0f;
    sx += gx_dps; sy += gy_dps; sz += gz_dps;
    valid++;
    delay(6); // ~166 Hz sampling during calibration
  }
  if (valid == 0) {
    Serial.println("[CAL] No valid samples - calibration failed");
    return;
  }
  gyroBiasX = sx / valid;
  gyroBiasY = sy / valid;
  gyroBiasZ = sz / valid;
  Serial.print("[CAL] Done. Biases (deg/s): ");
  Serial.print(gyroBiasX,4); Serial.print(", ");
  Serial.print(gyroBiasY,4); Serial.print(", ");
  Serial.println(gyroBiasZ,4);
}

// ---------------- EEPROM functions ----------------
void saveBiasesToEEPROM() {
  // mark flag and store floats
  EEPROM.update(EEPROM_FLAG_ADDR, 0xA5);
  // floats at addresses 4,8,12
  int addr = EEPROM_BIAS_ADDR;
  // write float bytes
  union { float f; uint8_t b[4]; } u;
  u.f = gyroBiasX;
  for (int i=0;i<4;i++) EEPROM.update(addr++, u.b[i]);
  u.f = gyroBiasY;
  for (int i=0;i<4;i++) EEPROM.update(addr++, u.b[i]);
  u.f = gyroBiasZ;
  for (int i=0;i<4;i++) EEPROM.update(addr++, u.b[i]);
  Serial.println("[EEPROM] Biases saved.");
}

bool loadBiasesFromEEPROM() {
  uint8_t flag = EEPROM.read(EEPROM_FLAG_ADDR);
  if (flag != 0xA5) return false;
  int addr = EEPROM_BIAS_ADDR;
  union { float f; uint8_t b[4]; } u;
  for (int i=0;i<4;i++) u.b[i] = EEPROM.read(addr++);
  gyroBiasX = u.f;
  for (int i=0;i<4;i++) u.b[i] = EEPROM.read(addr++);
  gyroBiasY = u.f;
  for (int i=0;i<4;i++) u.b[i] = EEPROM.read(addr++);
  gyroBiasZ = u.f;
  return true;
}

void resetBiases() {
  gyroBiasX = gyroBiasY = gyroBiasZ = 0.0f;
}

// ---------------- I2C helpers (robust read style) ----------------
bool writeRegister(uint8_t dev, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(dev);
  Wire.write(reg);
  Wire.write(val);
  uint8_t e = Wire.endTransmission();
  return (e == 0);
}

bool readRegisters(uint8_t dev, uint8_t startReg, uint8_t *buffer, uint8_t len, unsigned int timeoutMs) {
  Wire.beginTransmission(dev);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom((int)dev, (int)len);
  uint8_t idx = 0;
  unsigned long t0 = millis();
  while (idx < len && (millis() - t0) < timeoutMs) {
    if (Wire.available()) buffer[idx++] = Wire.read();
  }
  return (idx == len);
}

// ---------------- JSON output ----------------
void sendQuaternionJSON(unsigned long tmicros) {
  // Format: {"q":[x,y,z,w],"t":<micros>}
  // using q1,q2,q3,q0 ordering (x,y,z,w) to match earlier convention
  Serial.print("{\"q\":[");
  Serial.print(q1, 6); Serial.print(",");
  Serial.print(q2, 6); Serial.print(",");
  Serial.print(q3, 6); Serial.print(",");
  Serial.print(q0, 6); Serial.print("],\"t\":");
  Serial.print(tmicros);
  Serial.println("}");
}

// ---------------- Madgwick IMU-only algorithm ----------------
void MadgwickUpdateIMU(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
  // gx,gy,gz in rad/s; ax,ay,az in g; dt in s
  float recipNorm;
  float s0, s1, s2, s3;
  float qDot1, qDot2, qDot3, qDot4;
  float _2q0 = 2.0f * q0;
  float _2q1 = 2.0f * q1;
  float _2q2 = 2.0f * q2;
  float _2q3 = 2.0f * q3;
  float _4q0 = 4.0f * q0;
  float _4q1 = 4.0f * q1;
  float _4q2 = 4.0f * q2;
  float _8q1 = 8.0f * q1;
  float _8q2 = 8.0f * q2;
  float q0q0 = q0 * q0;
  float q1q1 = q1 * q1;
  float q2q2 = q2 * q2;
  float q3q3 = q3 * q3;

  // Rate of change from gyros
  qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
  qDot2 = 0.5f * ( q0 * gx + q2 * gz - q3 * gy);
  qDot3 = 0.5f * ( q0 * gy - q1 * gz + q3 * gx);
  qDot4 = 0.5f * ( q0 * gz + q1 * gy - q2 * gx);

  // Normalise accelerometer measurement
  float axn = ax, ayn = ay, azn = az;
  recipNorm = sqrt(axn * axn + ayn * ayn + azn * azn);
  if (recipNorm == 0.0f) return;
  recipNorm = 1.0f / recipNorm;
  axn *= recipNorm; ayn *= recipNorm; azn *= recipNorm;

  // Gradient descent corrective step
  s0 = _4q0 * q2q2 + _2q2 * axn + _4q0 * q1q1 - _2q1 * ayn;
  s1 = _4q1 * q3q3 - _2q3 * axn + 4.0f * q0q0 * q1 - _2q0 * ayn - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * azn;
  s2 = 4.0f * q0q0 * q2 + _2q0 * axn + _4q2 * q3q3 - _2q3 * ayn - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * azn;
  s3 = 4.0f * q1q1 * q3 - _2q1 * axn + 4.0f * q2q2 * q3 - _2q2 * ayn;

  recipNorm = sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
  if (recipNorm == 0.0f) return;
  recipNorm = 1.0f / recipNorm;
  s0 *= recipNorm; s1 *= recipNorm; s2 *= recipNorm; s3 *= recipNorm;

  // Apply feedback
  qDot1 -= beta * s0;
  qDot2 -= beta * s1;
  qDot3 -= beta * s2;
  qDot4 -= beta * s3;

  // Integrate
  q0 += qDot1 * dt;
  q1 += qDot2 * dt;
  q2 += qDot3 * dt;
  q3 += qDot4 * dt;

  // Normalize quaternion
  recipNorm = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  recipNorm = 1.0f / recipNorm;
  q0 *= recipNorm; q1 *= recipNorm; q2 *= recipNorm; q3 *= recipNorm;
}
