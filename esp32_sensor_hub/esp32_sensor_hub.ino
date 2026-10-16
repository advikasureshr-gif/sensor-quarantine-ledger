/*
  esp32_sensor_hub.ino - streams one JSON line per reading over USB serial (see iotguard/contract.py).

    {"t":12.5,"temp":27.0,"hum":61.0,"pir":1,"door":0,"ldr":1873}

  Board:    "ESP32 Dev Module", 115200 baud.   Library: "DHT sensor library" by Adafruit (+ Adafruit Unified Sensor).
  Wiring (matches the four-sensor test sketch):
    DHT11  data -> GPIO 27  (VCC 3.3 V, GND; many DHT11 modules have the pull-up built in)
    PIR    out  -> GPIO 13  (HC-SR501: VCC 5 V from VIN/5V pin, GND)
    Reed   one leg -> GPIO 14, other leg -> GND   (internal pull-up is used, no resistor needed)
    LDR    module DO -> GPIO 33   (LDR_DIGITAL = true: only 0/1 light information, so the LDR is a coarse bright/dark signal).
           If you later get AO or a bare LDR (+10k divider to GPIO 33), set LDR_DIGITAL = false for real brightness values.
  Use an ADC1 pin for the LDR (GPIO 32-39). ADC2 pins do not work reliably when WiFi is on.

  Keep SAMPLE_PERIOD_MS in step with SamplingConfig.sample_period_s in iotguard/config.py (0.5 s).
  Only one program can hold the serial port: close the Arduino Serial Monitor before running Python.
*/
#include <DHT.h

// ---- pins -----------------------------------------------------------------
const int PIN_DHT  = 27;
const int PIN_PIR  = 13;
const int PIN_DOOR = 14;
const int PIN_LDR  = 33;   // module DO for now (see LDR_DIGITAL below); GPIO 33 is also an ADC1 pin

// ---- settings --------------------------------------------------------------
const unsigned long SAMPLE_PERIOD_MS = 500;   // must match config.py
const bool DOOR_OPEN_IS_HIGH = true;          // reed to GND with pull-up: magnet near = closed = LOW, so open = HIGH
const int LDR_SAMPLES = 4;                    // averaged to tame ADC noise (analog mode)
const bool LDR_DIGITAL = true;                // true: the module only has DO (0/1). false: AO wired to the pin (real 0..4095)
const int LDR_DIGITAL_SAMPLES = 16;           // digital mode: DO is read this many times per reading, ~3 ms apart in total

DHT dht(PIN_DHT, DHT11);
unsigned long nextTick;

static int readLdr() {
  if (LDR_DIGITAL) {
    // DO is just a comparator output. Scale it onto the 0..4095 range the pipeline expects. Polarity does not matter:
    // the pipeline only looks at relative change. Averaging a few quick reads smooths flicker when the light sits right
    // at the module's threshold (turn the blue potentiometer on the module until that happens only rarely).
    int high = 0;
    for (int i = 0; i < LDR_DIGITAL_SAMPLES; i++) { high += digitalRead(PIN_LDR); delayMicroseconds(200); }
    return (int)((long)high * 4095 / LDR_DIGITAL_SAMPLES);
  }
  long sum = 0;
  for (int i = 0; i < LDR_SAMPLES; i++) sum += analogRead(PIN_LDR);   // 12-bit: 0..4095
  return (int)(sum / LDR_SAMPLES);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_PIR, INPUT);
  pinMode(PIN_LDR, INPUT);                      // module DO (digital) or AO (analog): a plain input either way
  pinMode(PIN_DOOR, INPUT_PULLUP);
  analogReadResolution(12);
  dht.begin();
  delay(1500);
  Serial.println("# esp32_sensor_hub ready");
  Serial.println("# note: HC-SR501 needs ~60 s after power-up before its output is meaningful");
  nextTick = millis();
}

void loop() {
  unsigned long now = millis();
  if ((long)(now - nextTick) < 0) return;
  nextTick += SAMPLE_PERIOD_MS;                 // fixed schedule: no drift
  if ((long)(now - nextTick) > (long)(5 * SAMPLE_PERIOD_MS)) nextTick = now + SAMPLE_PERIOD_MS;  // fell far behind: resync

  float h = dht.readHumidity();                 // the library rate-limits itself to the DHT11's ~1 s
  float t = dht.readTemperature();
  int pir  = digitalRead(PIN_PIR) ? 1 : 0;
  int door = (digitalRead(PIN_DOOR) == (DOOR_OPEN_IS_HIGH ? HIGH : LOW)) ? 1 : 0;
  int ldr  = readLdr();

  Serial.print("{\"t\":");
  Serial.print(now / 1000.0, 1);
  Serial.print(",\"temp\":");
  if (isnan(t) || isnan(h)) Serial.print("null"); else Serial.print(t, 1);
  Serial.print(",\"hum\":");
  if (isnan(t) || isnan(h)) Serial.print("null"); else Serial.print(h, 1);
  Serial.print(",\"pir\":");  Serial.print(pir);
  Serial.print(",\"door\":"); Serial.print(door);
  Serial.print(",\"ldr\":");  Serial.print(ldr);
  Serial.println("}");
}
