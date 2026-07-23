#include <Wire.h>

/*
  BH1750 minimal driver + demo (no external libraries)

  Wiring (typical):
    BH1750 VCC -> 3.3V or 5V (check your breakout)
    BH1750 GND -> GND
    BH1750 SDA -> SDA
    BH1750 SCL -> SCL
    ADDR pin (if present): LOW => 0x23, HIGH => 0x5C

  Notes:
    - BH1750 outputs 16-bit value. Lux = raw / 1.2 for typical modes.
    - Measurement time depends on mode; we use reasonable delays and also poll with a timeout.
*/

class BH1750Driver {
public:
  enum class Mode : uint8_t {
    // Continuous modes
    CONT_H_RES   = 0x10, // 1 lx resolution, ~120 ms
    CONT_H_RES2  = 0x11, // 0.5 lx resolution, ~120 ms
    CONT_L_RES   = 0x13, // 4 lx resolution, ~16 ms
    // One-time modes (power down after each measurement)
    ONE_H_RES    = 0x20,
    ONE_H_RES2   = 0x21,
    ONE_L_RES    = 0x23
  };

  BH1750Driver() = default;

  // If you know address, pass 0x23 or 0x5C. If you pass 0, it will auto-detect.
  bool begin(TwoWire& wire = Wire, uint8_t address = 0, Mode mode = Mode::CONT_H_RES) {
    _wire = &wire;
    _wire->begin();

    if (address == 0) {
      if (probe(0x23)) _addr = 0x23;
      else if (probe(0x5C)) _addr = 0x5C;
      else return false;
    } else {
      _addr = address;
      if (!probe(_addr)) return false;
    }

    // Power on + reset + set mode
    if (!writeCmd(0x01)) return false; // POWER_ON
    delay(10);
    if (!writeCmd(0x07)) return false; // RESET (only valid when powered on)
    delay(10);

    _mode = mode;
    if (!setMode(mode)) return false;

    return true;
  }

  bool setMode(Mode mode) {
    _mode = mode;
    return writeCmd(static_cast<uint8_t>(mode));
  }

  // Optional calibration: multiply lux by this factor (e.g. 1.05)
  void setCalibrationFactor(float factor) {
    if (factor <= 0.0f) factor = 1.0f;
    _cal = factor;
  }

  // Returns true if read succeeded; lux_out gets updated.
  // For one-time modes, this triggers a new measurement command each time.
  bool readLux(float& lux_out, uint32_t timeout_ms = 250) {
    if (_wire == nullptr) return false;

    // If one-shot mode, we must re-send the mode command each read
    if (isOneTime(_mode)) {
      if (!writeCmd(static_cast<uint8_t>(_mode))) return false;
    }

    // Wait the typical measurement time (plus some cushion), then read.
    // We'll do a short initial delay (mode-dependent) and then poll-read with timeout.
    delay(measurementDelayMs(_mode));

    uint16_t raw = 0;
    if (!readRaw(raw, timeout_ms)) return false;

    // Typical conversion factor for BH1750 modes:
    // lux = raw / 1.2  (datasheet default MTreg)
    float lux = (static_cast<float>(raw) / 1.2f) * _cal;
    lux_out = lux;
    return true;
  }

  // If you want raw value (before conversion)
  bool readRaw(uint16_t& raw_out, uint32_t timeout_ms = 250) {
    if (_wire == nullptr) return false;

    // Request 2 bytes
    uint32_t start = millis();
    while (true) {
      _wire->beginTransmission(_addr);
      uint8_t txStatus = _wire->endTransmission(false); // repeated start
      if (txStatus != 0) {
        // Device not responding
        if (millis() - start >= timeout_ms) return false;
        delay(5);
        continue;
      }

      int n = _wire->requestFrom(static_cast<int>(_addr), 2);
      if (n == 2) {
        uint8_t msb = _wire->read();
        uint8_t lsb = _wire->read();
        raw_out = (static_cast<uint16_t>(msb) << 8) | lsb;
        return true;
      }

      if (millis() - start >= timeout_ms) return false;
      delay(5);
    }
  }

  uint8_t address() const { return _addr; }

private:
  TwoWire* _wire = nullptr;
  uint8_t _addr = 0;
  Mode _mode = Mode::CONT_H_RES;
  float _cal = 1.0f;

  bool probe(uint8_t addr) {
    _wire->beginTransmission(addr);
    return (_wire->endTransmission() == 0);
  }

  bool writeCmd(uint8_t cmd) {
    _wire->beginTransmission(_addr);
    _wire->write(cmd);
    return (_wire->endTransmission() == 0);
  }

  static bool isOneTime(Mode m) {
    uint8_t v = static_cast<uint8_t>(m);
    return (v == 0x20 || v == 0x21 || v == 0x23);
  }

  static uint16_t measurementDelayMs(Mode m) {
    // Typical timings: H_RES/H_RES2 ~120ms, L_RES ~16ms
    // Add cushion for slow boards / I2C.
    switch (m) {
      case Mode::CONT_L_RES:
      case Mode::ONE_L_RES:
        return 24;
      case Mode::CONT_H_RES:
      case Mode::CONT_H_RES2:
      case Mode::ONE_H_RES:
      case Mode::ONE_H_RES2:
      default:
        return 180;
    }
  }
};

// -------------------- Demo --------------------

BH1750Driver light;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  // If you know your address, set it explicitly (0x23 or 0x5C).
  // Otherwise pass 0 to auto-detect.
  bool ok = light.begin(Wire, 0, BH1750Driver::Mode::CONT_H_RES);
  if (!ok) {
    Serial.println("BH1750 not found on I2C (checked 0x23 and 0x5C). Fix wiring/address.");
    while (true) { delay(1000); }
  }

  light.setCalibrationFactor(1.0f); // adjust if you have a reference lux meter
  Serial.print("BH1750 OK at address 0x");
  Serial.println(light.address(), HEX);
}

void loop() {
  float lux = 0.0f;
  if (light.readLux(lux)) {
    Serial.print("Lux: ");
    Serial.println(lux, 2);
  } else {
    Serial.println("Read failed (timeout/I2C).");
  }
  delay(250);
}
