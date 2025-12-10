// High-speed EMG streaming for Upside Down Minima
// Target ~500 Hz

const int EMG_PIN = A1;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int emg = analogRead(EMG_PIN);
  Serial.println(emg);

  // 2 ms delay -> ~500 samples per second
  delayMicroseconds(2000);
}