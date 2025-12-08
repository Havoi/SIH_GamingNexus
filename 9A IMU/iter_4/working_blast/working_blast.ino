/*
  ESP32-S3 + ICM-20948 (6-AXIS MODE - FIXED)
  ------------------------------------------
  Library: ICM20948_WE
  Pins: SDA=7, SCL=6
  Address: 0x69
*/

#include <Wire.h>
#include <ICM20948_WE.h>

#define SDA_PIN 7
#define SCL_PIN 6
#define ICM_ADDR 0x69

ICM20948_WE myIMU = ICM20948_WE(ICM_ADDR);

void setup() {
  Serial.begin(115200);
  
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000); 

  // Init Main Chip
  if (!myIMU.init()) {
    Serial.println("Error: ICM-20948 not found!");
    while (1);
  }
  
  // Setup Ranges for Gaming
  myIMU.setAccRange(ICM20948_ACC_RANGE_4G); 
  myIMU.setGyrRange(ICM20948_GYRO_RANGE_1000); 
  
  // FIXED: Set DLPF (Low Pass Filter) individually
  // This removes jitter/noise from your hand movements
  myIMU.setAccDLPF(ICM20948_DLPF_6); 
  myIMU.setGyrDLPF(ICM20948_DLPF_6);

  delay(100);
}

void loop() {
  xyzFloat acc, gyr;

  // Read Sensors
  myIMU.readSensor(); 
  
  // Get Values using the pointer method you verified earlier
  myIMU.getAccRawValues(&acc);
  myIMU.getGyrRawValues(&gyr);

  // Format: RAW:ax,ay,az,gx,gy,gz,0,0,0
  // Note: We send 0,0,0 for the Mag slots to keep the website happy
  Serial.print("RAW:");
  Serial.print(acc.x); Serial.print(",");
  Serial.print(acc.y); Serial.print(",");
  Serial.print(acc.z); Serial.print(",");
  Serial.print(gyr.x); Serial.print(",");
  Serial.print(gyr.y); Serial.print(",");
  Serial.print(gyr.z); Serial.print(",");
  Serial.println("0,0,0"); 

  delay(5); // ~200Hz
}