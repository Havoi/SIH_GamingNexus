/*
  imu_serial.ino
  ESP32-S3: read ICM20948, Madgwick AHRS, stream JSON over Serial (USB)
  Accepts simple JSON commands on Serial (newline-terminated)
  - Outputs lines like:
    {"type":"imu","quat":[w,x,y,z],"acc":[ax,ay,az],"gyro":[gx,gy,gz],"t":12345}
  - Accepts commands:
    {"cmd":"calibrate"}
    {"cmd":"set","k":"sensitivity","v":1.25}
    {"cmd":"calibrate_request"} (rare)
*/

#include <Wire.h>
#include <ICM20948_WE.h>

// ---------- USER PINS ----------
#define SDA_PIN 7
#define SCL_PIN 6
const int CAL_BUTTON_PIN = 0; // set to -1 if unused

// ---------- IMU constants ----------
ICM20948_WE myIMU = ICM20948_WE(0x69);
const float ACC_LSB_2G      = 16384.0f;
const float GYRO_LSB_250DPS = 131.0f;

// ---------- timing ----------
const float TARGET_HZ = 200.0f;
const unsigned long SAMPLE_US = (unsigned long)(1000000.0f / TARGET_HZ);
unsigned long lastSampleMicros = 0;

// ---------- Madgwick AHRS (embedded) ----------
class Madgwick {
public:
  float beta;
  float q0,q1,q2,q3;
  Madgwick(float b=0.12f) : beta(b), q0(1), q1(0), q2(0), q3(0) {}
  void update(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
    if (ax == 0.0f && ay == 0.0f && az == 0.0f) return;
    // Normalize accel
    float recipNorm = invSqrt(ax*ax + ay*ay + az*az);
    ax *= recipNorm; ay *= recipNorm; az *= recipNorm;
    // auxiliary variables
    float _2q0 = 2.0f*q0, _2q1 = 2.0f*q1, _2q2 = 2.0f*q2, _2q3 = 2.0f*q3;
    float _4q0 = 4.0f*q0, _4q1 = 4.0f*q1, _4q2 = 4.0f*q2, _8q1 = 8.0f*q1, _8q2 = 8.0f*q2;
    // gradient descent step
    float f1 = 2*(q1*q3 - q0*q2) - ax;
    float f2 = 2*(q0*q1 + q2*q3) - ay;
    float f3 = 2*(0.5f - q1*q1 - q2*q2) - az;
    float s0 = -_2q2 * f1 + _2q1 * f2;
    float s1 = _2q3 * f1 + _2q0 * f2 - _4q1 * f3;
    float s2 = -_2q0 * f1 + _2q3 * f2 - _4q2 * f3;
    float s3 = _2q1 * f1 + _2q2 * f2;
    recipNorm = invSqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3);
    s0 *= recipNorm; s1 *= recipNorm; s2 *= recipNorm; s3 *= recipNorm;
    // rate of change
    float qDot0 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz) - beta * s0;
    float qDot1 = 0.5f * ( q0 * gx + q2 * gz - q3 * gy) - beta * s1;
    float qDot2 = 0.5f * ( q0 * gy - q1 * gz + q3 * gx) - beta * s2;
    float qDot3 = 0.5f * ( q0 * gz + q1 * gy - q2 * gx) - beta * s3;
    // integrate
    q0 += qDot0 * dt;
    q1 += qDot1 * dt;
    q2 += qDot2 * dt;
    q3 += qDot3 * dt;
    recipNorm = invSqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
    q0 *= recipNorm; q1 *= recipNorm; q2 *= recipNorm; q3 *= recipNorm;
  }
  static inline float invSqrt(float x){ return 1.0f / sqrtf(x); }
};

Madgwick ahrs(0.12f);

// ---------- serial rx buffer ----------
String serialBuf = "";

// ---------- parameters forwarded (no effect on ESP logic but stored) ----------
float param_sensitivity = 1.0f;
float param_smoothing = 0.2f;
float param_deadzone = 0.02f;

// ---------- utilities ----------
inline float deg2rad(float d){ return d * 3.14159265358979323846f / 180.0f; }
inline float rad2deg(float r){ return r * 180.0f / 3.14159265358979323846f; }

// ---------- setup ----------
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000) ; // wait briefly for USB serial
  delay(50);
  pinMode(CAL_BUTTON_PIN, (CAL_BUTTON_PIN>=0)?INPUT_PULLUP:INPUT);
  // I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  Serial.println("{\"type\":\"status\",\"msg\":\"boot\"}");

  if (!myIMU.init()) {
    Serial.println("{\"type\":\"error\",\"msg\":\"IMU not detected\"}");
    while (1) { delay(1000); }
  }
  myIMU.setAccRange(ICM20948_ACC_RANGE_2G);
  myIMU.setGyrRange(ICM20948_GYRO_RANGE_250);

  lastSampleMicros = micros();
}

// ---------- simple non-blocking serial parse ----------
void handleSerialRx() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      String line = serialBuf;
      serialBuf = "";
      if (line.length() > 0) processSerialLine(line);
    } else {
      if (serialBuf.length() < 1024) serialBuf += c;
    }
  }
}

void processSerialLine(const String &line) {
  // very small parser looking for keys we expect
  // commands supported: {"cmd":"calibrate"} or {"cmd":"set","k":"sensitivity","v":1.2}
  String s = line;
  s.trim();
  if (s.indexOf("\"cmd\":\"calibrate\"") >= 0) {
    // ack back
    Serial.println("{\"type\":\"ack\",\"cmd\":\"calibrate\"}");
    // nothing else to do - python will handle baseline
  } else if (s.indexOf("\"cmd\":\"calibrate_request\"") >= 0) {
    Serial.println("{\"type\":\"ack\",\"cmd\":\"calibrate_request\"}");
  } else {
    // try "set" commands
    int idxK = s.indexOf("\"k\":\"");
    if (idxK >= 0) {
      int kstart = idxK + 5;
      int kend = s.indexOf('"', kstart);
      if (kend > kstart) {
        String key = s.substring(kstart, kend);
        int idxV = s.indexOf("\"v\":", kend);
        if (idxV > 0) {
          int vstart = idxV + 4;
          String vstr = s.substring(vstart);
          // strip non numeric
          int endpos = vstr.indexOf('}');
          if (endpos > 0) vstr = vstr.substring(0, endpos);
          float v = vstr.toFloat();
          if (key == "sensitivity") param_sensitivity = v;
          else if (key == "smoothing") param_smoothing = v;
          else if (key == "deadzone") param_deadzone = v;
          char ack[128];
          snprintf(ack, sizeof(ack), "{\"type\":\"param_ack\",\"k\":\"%s\",\"v\":%.4f}", key.c_str(), v);
          Serial.println(ack);
        }
      }
    }
  }
}

// ---------- main loop ----------
void loop() {
  // serial rx parse
  handleSerialRx();

  // check button press -> issue calibration request message
  static bool lastBtn = HIGH;
  if (CAL_BUTTON_PIN >= 0) {
    bool cur = digitalRead(CAL_BUTTON_PIN);
    if (cur == LOW && lastBtn == HIGH) {
      // button pressed
      Serial.println("{\"type\":\"cal_request\"}");
    }
    lastBtn = cur;
  }

  // timed sampling
  unsigned long now = micros();
  if (now - lastSampleMicros < SAMPLE_US) return;
  unsigned long dtMicros = now - lastSampleMicros;
  lastSampleMicros = now;
  float dt = dtMicros / 1000000.0f;
  if (dt <= 0 || dt > 0.1f) dt = 0.01f;

  // read IMU
  myIMU.readSensor();
  xyzFloat accRaw, gyrRaw;
  myIMU.getAccRawValues(&accRaw);
  myIMU.getGyrRawValues(&gyrRaw);

  // convert
  float ax = accRaw.x / ACC_LSB_2G;
  float ay = accRaw.y / ACC_LSB_2G;
  float az = accRaw.z / ACC_LSB_2G;
  float gx = (gyrRaw.x) / GYRO_LSB_250DPS; // deg/s
  float gy = (gyrRaw.y) / GYRO_LSB_250DPS;
  float gz = (gyrRaw.z) / GYRO_LSB_250DPS;
  float gx_r = deg2rad(gx), gy_r = deg2rad(gy), gz_r = deg2rad(gz);

  // update AHRS
  ahrs.update(gx_r, gy_r, gz_r, ax, ay, az, dt);

  // build JSON line
  char out[256];
  unsigned long tms = millis();
  snprintf(out, sizeof(out),
    "{\"type\":\"imu\",\"quat\":[%.6f,%.6f,%.6f,%.6f],\"acc\":[%.5f,%.5f,%.5f],\"gyro\":[%.4f,%.4f,%.4f],\"t\":%lu}",
    ahrs.q0, ahrs.q1, ahrs.q2, ahrs.q3,
    ax, ay, az,
    gx, gy, gz,
    tms
  );
  Serial.println(out);

  // loop back immediately (non-blocking)
}
