#include <Wire.h>
#include <ICM_20948.h> // SparkFun ICM-20948 Library

ICM_20948_I2C myICM;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000); // Fast I2C

  myICM.begin(Wire, 1);
  
  // Set Scale for higher precision if needed (defaults are usually fine for gaming)
  // myICM.setDSSS(ICM_20948_DSS_ENABLE); 
}

void loop() {
  if (myICM.dataReady()) {
    myICM.getAGMT(); // Read Acc, Gyro, Mag, Temp

    // Format: "RAW:ax,ay,az,gx,gy,gz,mx,my,mz"
    Serial.print("RAW:");
    Serial.print(myICM.accX()); Serial.print(",");
    Serial.print(myICM.accY()); Serial.print(",");
    Serial.print(myICM.accZ()); Serial.print(",");
    Serial.print(myICM.gyrX()); Serial.print(",");
    Serial.print(myICM.gyrY()); Serial.print(",");
    Serial.print(myICM.gyrZ()); Serial.print(",");
    Serial.print(myICM.magX()); Serial.print(",");
    Serial.print(myICM.magY()); Serial.print(",");
    Serial.println(myICM.magZ());
    
    delay(5); // ~200Hz update rate
  }
}