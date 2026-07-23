#include <ESP32-HUB75-MatrixPanel-I2S-DMA.h>
#include <FastLED.h>

// ------------------- Panel config -------------------
#define PANEL_WIDTH   64
#define PANEL_HEIGHT  64
#define PANELS_NUMBER 4
#define PIN_E         17

#define PANE_WIDTH  (PANEL_WIDTH * PANELS_NUMBER)  // 256
#define PANE_HEIGHT (PANEL_HEIGHT)                 // 64

MatrixPanel_I2S_DMA *dma_display = nullptr;

static const uint8_t R = 255, G = 255, B = 255;

// ------------------- Brightness ramp -------------------
// flux_comp expects a clean line like: BRIGHT=NN
static int bright_pct = 0;                 // 0..100 (%)
static const uint32_t STEP_DELAY_MS = 3000; // 3 seconds per 1%

// Convert 0–100% to 0–255
static inline uint8_t pct_to_255(int pct) {
  if (pct < 0) pct = 0;
  if (pct > 100) pct = 100;
  // Round-to-nearest integer
  return (uint8_t)((pct * 255 + 50) / 100);
}

void fill_white_fullpanel() {
  if (!dma_display) return;
  dma_display->fillScreenRGB888(R, G, B);
}

void setup() {
  Serial.begin(115200);

  HUB75_I2S_CFG mxconfig;
  mxconfig.mx_height = PANEL_HEIGHT;
  mxconfig.chain_length = PANELS_NUMBER;
  mxconfig.gpio.e = PIN_E;

  dma_display = new MatrixPanel_I2S_DMA(mxconfig);

  // Start at 0% brightness
  dma_display->setBrightness8(pct_to_255(bright_pct));

  if (!dma_display->begin()) {
    // Avoid extra prints; flux_comp only wants BRIGHT= lines,
    // but if begin fails the panel won't work anyway.
    return;
  }

  // Draw full white once; then only change brightness
  fill_white_fullpanel();

  // Print initial brightness in the exact parseable format
  Serial.print("BRIGHT=");
  Serial.println(bright_pct);
}

void loop() {
  // Wait 3 seconds at each brightness level
  delay(STEP_DELAY_MS);

  // Step brightness 0->100 by 1, then wrap to 0
  bright_pct += 1;
  if (bright_pct > 100) bright_pct = 0;

  // Apply brightness (0..255)
  dma_display->setBrightness8(pct_to_255(bright_pct));

  // Print only brightness in format flux_comp can read
  Serial.print("BRIGHT=");
  Serial.println(bright_pct);
}
