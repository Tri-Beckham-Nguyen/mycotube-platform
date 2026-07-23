/*
  ESP32 HUB75 (4x 64x64 panels chained horizontally -> 256x64)

  Protocol (matches your Python):
    - STOP
    - SET [[r0,r1,r2,...], I, [on_s, off_s]]

  Fixes ghost/scattered pixels:
    - Uses double buffering
    - Draws to back buffer, then flips atomically
    - Hard-clears with fillScreenRGB888(0,0,0)
*/

#include <ESP32-HUB75-MatrixPanel-I2S-DMA.h>
#include <ArduinoJson.h>

// -------------------- Panel config --------------------
#define PANEL_WIDTH   64
#define PANEL_HEIGHT  64
#define PANELS_NUMBER 4
#define PIN_E         17

#define PANE_WIDTH  (PANEL_WIDTH * PANELS_NUMBER)  // 256
#define PANE_HEIGHT (PANEL_HEIGHT)                 // 64

static MatrixPanel_I2S_DMA *dma_display = nullptr;

// -------------------- Runtime state --------------------
static bool     pi_connected = false;
static uint32_t last_wait_print_ms = 0;

// Current pattern (repeats until replaced)
static int      bars_x[512];
static size_t   bars_n = 0;
static uint8_t  brightness8 = 0;
static uint32_t on_ms  = 1000;
static uint32_t off_ms = 1000;

// Blink FSM
static bool     phase_on = true;
static uint32_t phase_start_ms = 0;

// Serial line buffer
static char     linebuf[2048];
static size_t   line_len = 0;

// -------------------- Helpers --------------------
static uint32_t seconds_to_ms(double s) {
  if (s <= 0.0) return 0;
  double ms = s * 1000.0;
  if (ms > 4294967295.0) return 4294967295UL;
  return (uint32_t)(ms + 0.5); // round
}

static void hard_clear_backbuffer() {
  // Draw into the "back" buffer (library handles which is back/front)
  dma_display->fillScreenRGB888(0, 0, 0);
}

static void present_frame() {
  // Atomically swap buffers so the visible frame is coherent
  dma_display->flipDMABuffer();
}

static void draw_bars_frame(bool on) {
  // Always draw the entire frame into the back buffer, then flip.
  hard_clear_backbuffer();

  if (on && bars_n > 0 && brightness8 > 0) {
    // brightness is global; set before drawing
    dma_display->setBrightness8(brightness8);

    // White line color
    const uint16_t color = dma_display->color565(255, 255, 255);

    for (size_t i = 0; i < bars_n; i++) {
      int x = bars_x[i];
      if (x < 0 || x >= PANE_WIDTH) continue;

      // Safer than drawFastVLine on some panel/driver combos:
      // width=1, height=PANE_HEIGHT
      dma_display->fillRect(x, 0, 1, PANE_HEIGHT, color);
    }
  } else {
    dma_display->setBrightness8(0);
  }

  present_frame();
}

static void restart_cycle() {
  phase_on = true;
  phase_start_ms = millis();
  draw_bars_frame(true);
}

static void apply_stop() {
  bars_n = 0;
  brightness8 = 0;
  on_ms = 1000;
  off_ms = 1000;
  phase_on = false;
  phase_start_ms = millis();
  draw_bars_frame(false);
}

static bool parse_and_apply_set_payload(const char *payload) {
  // payload is expected to be JSON: [[...], I, [on, off]]
  StaticJsonDocument<4096> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) return false;

  if (!doc.is<JsonArray>()) return false;
  JsonArray top = doc.as<JsonArray>();
  if (top.size() != 3) return false;

  // r array
  if (!top[0].is<JsonArray>()) return false;
  JsonArray r = top[0].as<JsonArray>();

  // I
  double I_d = top[1].as<double>();
  int I = (int)(I_d + 0.5);
  if (I < 0) I = 0;
  if (I > 100) I = 100;

  // S array
  if (!top[2].is<JsonArray>()) return false;
  JsonArray S = top[2].as<JsonArray>();
  if (S.size() != 2) return false;

  double on_s  = S[0].as<double>();
  double off_s = S[1].as<double>();
  uint32_t new_on_ms  = seconds_to_ms(on_s);
  uint32_t new_off_ms = seconds_to_ms(off_s);

  // Store bars (dedupe, preserve order)
  bool used[PANE_WIDTH];
  for (int i = 0; i < PANE_WIDTH; i++) used[i] = false;

  size_t n = 0;
  for (JsonVariant v : r) {
    if (n >= (sizeof(bars_x) / sizeof(bars_x[0]))) break;
    if (!v.is<int>() && !v.is<float>() && !v.is<double>()) continue;

    int x = (int)v.as<double>();
    x %= PANE_WIDTH;
    if (x < 0) x += PANE_WIDTH;

    if (used[x]) continue;
    used[x] = true;
    bars_x[n++] = x;
  }
  bars_n = n;

  // brightness8
  uint32_t b = (uint32_t)((255.0 * (double)I / 100.0) + 0.5);
  if (b > 255) b = 255;
  brightness8 = (uint8_t)b;

  // timings
  on_ms  = new_on_ms;
  off_ms = new_off_ms;

  restart_cycle();
  return true;
}

static void handle_line(const char *line) {
  // Trim leading spaces
  while (*line == ' ' || *line == '\t') line++;

  if (strcmp(line, "STOP") == 0) {
    apply_stop();
    return;
  }

  // Accept "SET " prefix
  if (strncmp(line, "SET", 3) == 0) {
    line += 3;
    while (*line == ' ' || *line == '\t') line++;
    if (parse_and_apply_set_payload(line)) {
      if (!pi_connected) {
        pi_connected = true;
        Serial.println("ESP_READY");
      }
    }
    return;
  }

  // Also allow raw JSON (optional)
  if (line[0] == '[') {
    if (parse_and_apply_set_payload(line)) {
      if (!pi_connected) {
        pi_connected = true;
        Serial.println("ESP_READY");
      }
    }
    return;
  }

  // Optional handshake
  if (strcmp(line, "PI_HELLO") == 0) {
    pi_connected = true;
    Serial.println("ESP_READY");
    return;
  }
}

static void service_serial() {
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c < 0) break;
    if (c == '\r') continue;

    if (c == '\n') {
      linebuf[line_len] = '\0';
      handle_line(linebuf);
      line_len = 0;
      continue;
    }

    if (line_len + 1 < sizeof(linebuf)) {
      linebuf[line_len++] = (char)c;
    } else {
      // overflow -> drop line
      line_len = 0;
    }
  }
}

static void service_wait_message() {
  if (pi_connected) return;
  uint32_t now = millis();
  if (now - last_wait_print_ms >= 5000) {
    last_wait_print_ms = now;
    Serial.println("Wait for Pi connection");
    Serial.println("ESP_PING");
  }
}

static void service_blink_fsm() {
  uint32_t now = millis();

  if (phase_on) {
    if (on_ms == 0) {
      phase_on = false;
      phase_start_ms = now;
      draw_bars_frame(false);
      return;
    }
    if (now - phase_start_ms >= on_ms) {
      phase_on = false;
      phase_start_ms = now;
      draw_bars_frame(false);
    }
  } else {
    if (off_ms == 0) {
      phase_on = true;
      phase_start_ms = now;
      draw_bars_frame(true);
      return;
    }
    if (now - phase_start_ms >= off_ms) {
      phase_on = true;
      phase_start_ms = now;
      draw_bars_frame(true);
    }
  }
}

// -------------------- Arduino setup/loop --------------------
void setup() {
  Serial.begin(115200);

  HUB75_I2S_CFG mxconfig;
  mxconfig.mx_height    = PANEL_HEIGHT;
  mxconfig.chain_length = PANELS_NUMBER;
  mxconfig.gpio.e       = PIN_E;

  // IMPORTANT: coherent frames
  mxconfig.double_buff = true;

  // If you still see artifacts later, try lowering this:
  // mxconfig.i2sspeed = HUB75_I2S_CFG::HZ_10M;

  // If your panels are FM6126A-based and look glitchy, try:
  // mxconfig.driver = HUB75_I2S_CFG::FM6126A;

  dma_display = new MatrixPanel_I2S_DMA(mxconfig);

  if (!dma_display->begin()) {
    Serial.println("I2S DMA begin failed (memory allocation?)");
  }

  // Start fully black on both buffers
  dma_display->setBrightness8(0);
  hard_clear_backbuffer();
  present_frame();
  hard_clear_backbuffer();
  present_frame();

  apply_stop();
}

void loop() {
  service_wait_message();
  service_serial();
  service_blink_fsm();
}