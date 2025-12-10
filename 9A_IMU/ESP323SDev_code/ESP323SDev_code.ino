#include <Wire.h>
#include <ICM20948_WE.h>

#define SDA_PIN 7
#define SCL_PIN 6

ICM20948_WE myIMU = ICM20948_WE(0x69);  // your I2C address

// ==== Calibration & filter state ====
float gyroOffsetX = 0, gyroOffsetY = 0, gyroOffsetZ = 0;

float pitch = 0.0f;  // rotation around X (looking up/down)
float roll  = 0.0f;  // rotation around Y (tilt)
float yaw   = 0.0f;  // rotation around Z (looking left/right)

unsigned long lastTimeMicros = 0;

// Sensitivity for your chosen ranges:
const float ACC_LSB_2G       = 16384.0f;  // LSB/g for ±2g
const float GYRO_LSB_250DPS  = 131.0f;    // LSB/(deg/s) for ±250 dps

// Complementary filter coefficient
const float ALPHA = 0.98f;  // higher = more gyro, lower = more accel

void calibrateGyro(int samples = 500) {
  Serial.println("Calibrating gyro... Keep IMU still!");

  float sumX = 0, sumY = 0, sumZ = 0;
  xyzFloat g;

  for (int i = 0; i < samples; i++) {
    myIMU.readSensor();
    myIMU.getGyrRawValues(&g);
    sumX += g.x;
    sumY += g.y;
    sumZ += g.z;
    delay(2);
  }

  gyroOffsetX = sumX / samples;
  gyroOffsetY = sumY / samples;
  gyroOffsetZ = sumZ / samples;

  Serial.print("Gyro offsets: ");
  Serial.print(gyroOffsetX); Serial.print(", ");
  Serial.print(gyroOffsetY); Serial.print(", ");
  Serial.println(gyroOffsetZ);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);  // 100 kHz is very safe on ESP32-S3

  Serial.println("Initializing ICM-20948...");

  if (!myIMU.init()) {
    Serial.println("IMU NOT detected!");
    while (1);
  }

  Serial.println("ICM-20948 READY!");

  myIMU.setAccRange(ICM20948_ACC_RANGE_2G);
  myIMU.setGyrRange(ICM20948_GYRO_RANGE_250);

  calibrateGyro();

  // Initialize time
  lastTimeMicros = micros();

  // Initialize pitch/roll from accelerometer so filter starts close to reality
  xyzFloat acc;
  myIMU.readSensor();
  myIMU.getAccRawValues(&acc);

  float ax = acc.x / ACC_LSB_2G;
  float ay = acc.y / ACC_LSB_2G;
  float az = acc.z / ACC_LSB_2G;

  pitch = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / PI;
  roll  = atan2f(ay, az) * 180.0f / PI;
  yaw   = 0;  // start yaw at 0 (relative)
}

void loop() {
  // --- Time step (dt in seconds) ---
  unsigned long now = micros();
  float dt = (now - lastTimeMicros) / 1000000.0f;
  lastTimeMicros = now;
  if (dt <= 0 || dt > 0.1f) dt = 0.01f;  // clamp dt for safety

  // --- Read sensors ---
  myIMU.readSensor();

  xyzFloat accRaw, gyrRaw;
  myIMU.getAccRawValues(&accRaw);
  myIMU.getGyrRawValues(&gyrRaw);

  // Convert raw accel to g
  float ax = accRaw.x / ACC_LSB_2G;
  float ay = accRaw.y / ACC_LSB_2G;
  float az = accRaw.z / ACC_LSB_2G;

  // Convert raw gyro to deg/s and apply offsets
  float gx = (gyrRaw.x - gyroOffsetX) / GYRO_LSB_250DPS;
  float gy = (gyrRaw.y - gyroOffsetY) / GYRO_LSB_250DPS;
  float gz = (gyrRaw.z - gyroOffsetZ) / GYRO_LSB_250DPS;

  // --- Accel-only angles (degrees) ---
  float accPitch = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / PI;
  float accRoll  = atan2f(ay, az) * 180.0f / PI;

  // --- Gyro integration (degrees) ---
  // integrate rates into angles
  float pitchGyro = pitch + gy * dt;
  float rollGyro  = roll  + gx * dt;
  float yawGyro   = yaw   + gz * dt;

  // --- Complementary filter ---
  pitch = ALPHA * pitchGyro + (1.0f - ALPHA) * accPitch;
  roll  = ALPHA * rollGyro  + (1.0f - ALPHA) * accRoll;
  yaw   = yawGyro;  // for now yaw is only from gyro → good for relative camera

  // Optional: keep yaw bounded
  if (yaw > 180.0f)  yaw -= 360.0f;
  if (yaw < -180.0f) yaw += 360.0f;

  // --- Output: ready for game camera ---
  // Format = YPR: yaw pitch roll
  Serial.print("YPR: ");
  Serial.print(yaw);   Serial.print(" ");
  Serial.print(pitch); Serial.print(" ");
  Serial.println(roll);

  // If you still want raw debug, uncomment:
  /*
  Serial.print("RAW GYR (dps): ");
  Serial.print(gx); Serial.print(" ");
  Serial.print(gy); Serial.print(" ");
  Serial.println(gz);
  */

  delay(5);  // ~200 Hz updates
}