/*
  imu_udp_json.ino
  Reads ICM20948 via ICM20948_WE, computes Madgwick AHRS, sends JSON over UDP.
  Matches the JSON format:
  {"type":"imu","quat":[q0,q1,q2,q3],"acc":[ax,ay,az],"gyro":[gx,gy,gz],"t":12345}

  Configure:
    const char* ssid       = "HP 7267";
    const char* password   = "123456789";
    const char* udpAddress = "10.144.245.147";
    const int   udpPort    = 4210;
*/

#include <Wire.h>
#include <ICM20948_WE.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// ---------------- Wi-Fi / UDP ----------------
const char* ssid        = "116757";
const char* password    = "password";
const char* udpAddress  = "192.168.137.1";
const int   udpPort     = 4212;

WiFiUDP udp;

// ---------------- ICM20948 ----------------
ICM20948_WE imu = ICM20948_WE(0x69);
const float ACC_LSB_2G      = 16384.0f; // accel LSB for ±2g
const float GYRO_LSB_250DPS = 131.0f;  // gyro LSB for ±250 dps

#define SDA_PIN 7
#define SCL_PIN 6

// ---------------- Madgwick AHRS ----------------
class Madgwick {
public:
  float beta;
  float q0,q1,q2,q3;
  Madgwick(float b=0.12f) : beta(b), q0(1), q1(0), q2(0), q3(0) {}
  void update(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
    if (ax == 0.0f && ay == 0.0f && az == 0.0f) return;
    // normalize accelerometer
    float recipNorm = invSqrt(ax*ax + ay*ay + az*az);
    ax *= recipNorm; ay *= recipNorm; az *= recipNorm;
    // auxiliary variables
    float _2q0 = 2.0f*q0, _2q1 = 2.0f*q1, _2q2 = 2.0f*q2, _2q3 = 2.0f*q3;
    float _4q0 = 4.0f*q0, _4q1 = 4.0f*q1, _4q2 = 4.0f*q2;
    // objective function
    float f1 = 2*(q1*q3 - q0*q2) - ax;
    float f2 = 2*(q0*q1 + q2*q3) - ay;
    float f3 = 2*(0.5f - q1*q1 - q2*q2) - az;
    float s0 = -_2q2 * f1 + _2q1 * f2;
    float s1 =  _2q3 * f1 + _2q0 * f2 - _4q1 * f3;
    float s2 = -_2q0 * f1 + _2q3 * f2 - _4q2 * f3;
    float s3 =  _2q1 * f1 + _2q2 * f2;
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
    // normalize quaternion
    recipNorm = invSqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3);
    q0 *= recipNorm; q1 *= recipNorm; q2 *= recipNorm; q3 *= recipNorm;
  }
  static inline float invSqrt(float x){ return 1.0f / sqrtf(x); }
};
Madgwick ahrs(0.12f);

// ---------------- utilities ----------------
inline float deg2rad(float d){ return d * 3.14159265358979323846f / 180.0f; }
inline float rad2deg(float r){ return r * 180.0f / 3.14159265358979323846f; }

unsigned long lastMicros = 0;

// ---------------- setup ----------------
void setup() {
  Serial.begin(115200);
  delay(50);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println("{\"type\":\"status\",\"msg\":\"boot\"}");

  if (!imu.init()) {
    Serial.println("{\"type\":\"error\",\"msg\":\"IMU not detected\"}");
    while (1) { delay(1000); }
  }
  imu.setAccRange(ICM20948_ACC_RANGE_2G);
  imu.setGyrRange(ICM20948_GYRO_RANGE_250);
  lastMicros = micros();

  // WiFi
  Serial.print("Connecting to WiFi ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < 10000) {
    delay(200);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.printf("{\"type\":\"status\",\"msg\":\"wifi:connected\",\"ip\":\"%s\"}\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println();
    Serial.println("{\"type\":\"status\",\"msg\":\"wifi:not_connected\"}");
  }
  udp.begin(udpPort); // local bind (optional)
  Serial.printf("{\"type\":\"status\",\"msg\":\"udp_bound\",\"port\":%d}\n", udpPort);
}

// ---------------- main loop ----------------
void loop() {
  // sample at ~200 Hz if possible
  const float TARGET_HZ = 200.0f;
  const unsigned long SAMPLE_US = (unsigned long)(1000000.0f / TARGET_HZ);
  unsigned long now = micros();
  if (now - lastMicros < SAMPLE_US) return;
  unsigned long dtMicros = now - lastMicros;
  lastMicros = now;
  float dt = dtMicros / 1000000.0f;
  if (dt <= 0 || dt > 0.1f) dt = 0.01f;

  // read raw accel & gyro
  xyzFloat accRaw, gyrRaw;
  imu.readSensor();
  imu.getAccRawValues(&accRaw);
  imu.getGyrRawValues(&gyrRaw);

  // convert to physical units
  float ax = accRaw.x / ACC_LSB_2G; // g
  float ay = accRaw.y / ACC_LSB_2G;
  float az = accRaw.z / ACC_LSB_2G;
  float gx = (gyrRaw.x) / GYRO_LSB_250DPS; // deg/s
  float gy = (gyrRaw.y) / GYRO_LSB_250DPS;
  float gz = (gyrRaw.z) / GYRO_LSB_250DPS;

  // update AHRS: Madgwick expects gyro in rad/s
  ahrs.update(deg2rad(gx), deg2rad(gy), deg2rad(gz), ax, ay, az, dt);

  // build JSON (same formatting as you used previously)
  char out[256];
  unsigned long tms = millis();
  snprintf(out, sizeof(out),
    "{\"type\":\"imu\",\"quat\":[%.6f,%.6f,%.6f,%.6f],\"acc\":[%.5f,%.5f,%.5f],\"gyro\":[%.4f,%.4f,%.4f],\"t\":%lu}",
    ahrs.q0, ahrs.q1, ahrs.q2, ahrs.q3,
    ax, ay, az,
    gx, gy, gz,
    tms
  );

  // print to serial as well
  Serial.println(out);

  // send UDP (destination = udpAddress:udpPort)
  udp.beginPacket(udpAddress, udpPort);
  udp.write((const uint8_t*)out, strlen(out));
  udp.endPacket();
}
