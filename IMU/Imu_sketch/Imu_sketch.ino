/*
  Comprehensive MPU6050 / I2C diagnostic
  - Intended for Arduino Nano R4 Minima (but generic Wire usage)
  - Tries multiple I2C initializations (Wire.begin(), also common pin pairs)
  - Performs I2C scan, low-level ACK bitbang test, and bitbang WHO_AM_I read (0x75)
  - Prints clear results for us to act on.

  IMPORTANT:
  - Tie AD0 -> GND (force 0x68) before testing.
  - Use 3.3V for VCC unless your breakout explicitly accepts 5V.
*/

#include <Wire.h>

#define SERIAL_BAUD 115200

// Common addresses for MPU6050
const uint8_t ADDR1 = 0x68;
const uint8_t ADDR2 = 0x69;
const uint8_t WHO_AM_I_REG = 0x75;

// Candidate pin pairs to try (edit if you know the board's pins)
struct Pair { int sda; int scl; const char *name; };
Pair candidates[] = {
  { -1, -1, "Wire.begin() default pins" },   // default
  { A4, A5, "A4/A5 (classic Arduino I2C pins)" }, // common on Nano-style boards
  { 8, 9,  "GPIO 8/9 (you mentioned earlier)" },
  { 21, 22, "GPIO 21/22 (ESP boards; harmless to try on Nano)" }
};
const int N_CAND = sizeof(candidates)/sizeof(candidates[0]);

// Low-level bitbang pins (you can edit if you want)
int BB_SDA = A4; // will be overridden per candidate when bitbang test is run
int BB_SCL = A5;

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) { delay(5); } // wait for Serial on some boards
  Serial.println("\n=== MPU6050 COMPREHENSIVE DIAGNOSTIC ===");
  Serial.println("Make sure AD0 -> GND, VCC -> 3.3V, GND -> GND.");
  Serial.println("This will try multiple pin pairs & tests. Output follows.\n");
  delay(200);
  runAllTests();
  Serial.println("\n=== DIAGNOSTIC RUN COMPLETE ===");
  Serial.println("If no MPU found, paste all output here and I'll interpret.");
}

void loop() {
  // nothing — one-shot diagnostics in setup
  delay(1000);
}

/* ---------------- TEST SEQUENCE ---------------- */

void runAllTests() {
  for (int i=0;i<N_CAND;i++) {
    int sda = candidates[i].sda;
    int scl = candidates[i].scl;
    Serial.print("\n--- TRY: "); Serial.print(candidates[i].name); Serial.println(" ---");

    if (sda == -1 && scl == -1) {
      Serial.println("Calling Wire.begin() with default pins.");
      Wire.end();
      Wire.begin();
    } else {
      Serial.print("Calling Wire.begin("); Serial.print(sda); Serial.print(", "); Serial.print(scl); Serial.println(")");
      Wire.end();
      // Note: on some boards A4/A5 are constants; on others they map to numbers; safe to call.
      Wire.begin(sda, scl);
    }
    delay(50);

    Serial.println("-> I2C scan @ 100kHz:");
    Wire.setClock(100000);
    scanI2C();

    Serial.println("-> I2C scan @ 400kHz:");
    Wire.setClock(400000);
    scanI2C();

    // Try the standard Wire-based WHO_AM_I read for 0x68/0x69
    Serial.println("-> Wire-based WHO_AM_I probe:");
    probeWhoAmI(ADDR1);
    probeWhoAmI(ADDR2);

    // Low-level ACK bitbang test on the same SDA/SCL pins (if they are numeric)
    if (sda != -1 && scl != -1) {
      BB_SDA = sda;
      BB_SCL = scl;
      Serial.print("-> Low-level ACK bitbang test on pins SDA="); Serial.print(BB_SDA);
      Serial.print(" SCL="); Serial.println(BB_SCL);
      lowLevelAckBitbang(ADDR1);
      lowLevelAckBitbang(ADDR2);

      Serial.print("-> Bitbang WHO_AM_I read (pins SDA=");
      Serial.print(BB_SDA); Serial.print(" SCL="); Serial.print(BB_SCL); Serial.println(")");
      bitbangWHOAMI();
    } else {
      Serial.println("-> Skipping bitbang tests for default pins (unknown mapping).");
    }

    Serial.println("--- end of this candidate ---\n");
    delay(200);
  }
}

/* ---------- Wire helpers ---------- */
void scanI2C() {
  int found = 0;
  for (uint8_t addr=1; addr<127; ++addr) {
    Wire.beginTransmission(addr);
    uint8_t e = Wire.endTransmission();
    if (e == 0) {
      Serial.print("  FOUND device at 0x"); if (addr < 16) Serial.print('0'); Serial.println(addr, HEX);
      found++;
    } else if (e == 4) {
      Serial.print("  UNKNOWN error at 0x"); if (addr < 16) Serial.print('0'); Serial.println(addr, HEX);
    }
  }
  if (!found) Serial.println("  (no devices found)");
  else Serial.print("  total found: "), Serial.println(found);
}

void probeWhoAmI(uint8_t addr) {
  Serial.print("  Probe 0x"); Serial.print(addr, HEX); Serial.print(": ");
  Wire.beginTransmission(addr);
  Wire.write(WHO_AM_I_REG);
  uint8_t e = Wire.endTransmission(false); // restart
  if (e != 0) {
    Serial.print("endTransmission code="); Serial.println(e);
    return;
  }
  Wire.requestFrom((int)addr, 1);
  if (Wire.available()) {
    uint8_t who = Wire.read();
    Serial.print("WHO_AM_I = 0x"); Serial.println(who, HEX);
  } else {
    Serial.println("no data available (NACK or timeout)");
  }
}

/* ---------- Low-level bitbang ACK test ---------- */

void sclHigh() { pinMode(BB_SCL, OUTPUT); digitalWrite(BB_SCL, HIGH); delayMicroseconds(6); }
void sclLow()  { pinMode(BB_SCL, OUTPUT); digitalWrite(BB_SCL, LOW);  delayMicroseconds(6); }

// SDA output high/low or input pullup
void sdaOut(bool high) { pinMode(BB_SDA, OUTPUT); digitalWrite(BB_SDA, high?HIGH:LOW); delayMicroseconds(3); }
void sdaInput() { pinMode(BB_SDA, INPUT_PULLUP); delayMicroseconds(3); }
int  sdaRead() { pinMode(BB_SDA, INPUT_PULLUP); delayMicroseconds(3); return digitalRead(BB_SDA); }

void i2cStartBB() {
  sdaOut(true); sclHigh(); delayMicroseconds(4);
  sdaOut(false); delayMicroseconds(4);
  sclLow(); delayMicroseconds(4);
}

void i2cSendBitBB(int b) {
  sdaOut(b);
  delayMicroseconds(3);
  sclHigh();
  delayMicroseconds(6);
  sclLow();
}

int i2cReadBitBB() {
  sdaInput();
  delayMicroseconds(3);
  sclHigh();
  delayMicroseconds(6);
  int v = sdaRead();
  sclLow();
  return v;
}

void i2cSendByteBB(uint8_t b) {
  for (int i=7;i>=0;--i) i2cSendBitBB( (b>>i) & 1 );
}

void lowLevelAckBitbang(uint8_t addr) {
  i2cStartBB();
  uint8_t ab = (addr << 1) | 0; // write
  i2cSendByteBB(ab);
  int ack = i2cReadBitBB();
  Serial.print("    Address 0x"); Serial.print(addr, HEX); Serial.print(" -> ACK bit = ");
  Serial.println(ack); // 0 means ACK, 1 means NACK
  // stop
  sdaOut(false);
  sclHigh();
  delayMicroseconds(4);
  sdaOut(true);
  delayMicroseconds(4);
}

/* ---------- Bitbang WHO_AM_I register read ---------- */

void bitbangWHOAMI() {
  // We'll do: start, write (addr<<1|0), write reg(0x75), restart, write (addr<<1|1), read byte
  for (uint8_t addr : {ADDR1, ADDR2}) {
    Serial.print("    Addr 0x"); Serial.print(addr, HEX); Serial.print(": ");
    i2cStartBB();
    i2cSendByteBB((addr<<1)|0);
    int ack1 = i2cReadBitBB();
    Serial.print("after addr(write) ack="); Serial.print(ack1);
    if (ack1 != 0) {
      Serial.println(" (NACK) -> skipping");
      i2cStopBB();
      continue;
    }
    i2cSendByteBB(WHO_AM_I_REG);
    int ack2 = i2cReadBitBB();
    Serial.print(" regWriteAck="); Serial.print(ack2);
    // restart
    i2cStartBB();
    i2cSendByteBB((addr<<1)|1);
    int ack3 = i2cReadBitBB();
    Serial.print(" addr(read) ack="); Serial.print(ack3);
    if (ack3 != 0) {
      Serial.println(" (NACK on read)");
      i2cStopBB();
      continue;
    }
    uint8_t val = 0;
    // read 8 bits MSB first
    for (int i=7;i>=0;--i) {
      int bit = i2cReadBitBB();
      val |= (bit<<i);
    }
    // send NACK (1) to finish
    i2cSendBitBB(1);
    i2cStopBB();
    Serial.print(" WHO_AM_I=0x"); Serial.println(val, HEX);
  }
}

void i2cStopBB() {
  sdaOut(false);
  sclHigh();
  delayMicroseconds(4);
  sdaOut(true);
  delayMicroseconds(4);
}
