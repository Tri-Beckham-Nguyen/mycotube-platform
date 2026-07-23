# MycoTube Platform — instrumentation & control

> **Scope note.** This is a development archive, not a deployed release. It
> collects the instrumentation and control code I wrote for the Mycobotic
> project in Purdue's ARIES Lab. The files span more than one generation of the
> rig, and modules under `experiments/` target an earlier hardware
> configuration, so the repository is **not guaranteed to run end-to-end as a
> single unit**. The authoritative version runs on the lab's Raspberry Pi 5 and
> has not been reconciled with this repo. Published with the lab's permission.

The MycoTube is a fungal substrate instrumented with four differential electrode
pairs, surrounded by a 256-column addressable light field. The system applies a
controlled light stimulus, measures the substrate's bioelectrical response, and
logs both in sync so the relationship can be analyzed.

---

## Hardware

| Role | Part | Interface |
|------|------|-----------|
| Main controller | Raspberry Pi 5 | — |
| Signal acquisition | TI ADS1263 (32-bit ΔΣ ADC), 4 differential pairs | SPI |
| Light stimulus | ESP32 driving 4× chained 64×64 HUB75 panels (256×64) | USB serial |
| Light stimulus (earlier) | Arduino, mask-addressed panels | USB serial |
| Ambient light sensing | TSL2591 (Pi, I²C), BH1750 (Arduino, I²C) | I²C |
| Sensing | 4 stainless-steel differential electrode pairs | — |

Channel map (ADC1 differential): `diff1 = AIN0-1`, `diff2 = AIN2-3`,
`diff3 = AIN4-5`, `diff4 = AIN6-7`. Electrode walls sit at columns 32, 96, 160,
and 224 on the 256-column ring.

## The main loop

`pi5/mycotube_runner.py` is the experiment orchestrator. Python is the master
clock; the ESP32 is a display slave. Every **50 ms**, the runner:

1. Computes the light pattern — bar positions, intensity, blink timing — for the
   selected task.
2. Sends it to the ESP32: `SET [[positions], intensity, [on_s, off_s]]`.
3. Reads all four differential voltages from the ADS1263.
4. Writes one synchronized CSV row (stimulus state + four voltages) and updates
   a live plot at 10 Hz.

Four experiment tasks are implemented: a single-wall brightness ramp, a flashing
bar across configurable panel combinations, a rotating bar at selectable speed,
and a pattern-regeneration mode (triangle or diminishing-bars).

Light position is tracked as a **circular mean** on the 0–255 ring so patterns
wrap seamlessly across the panel seam. During OFF phases telemetry keeps
recording, intensity is forced to zero, and the last position is held rather
than reset — so an OFF interval is a real measurement condition, not a gap.

## Repository layout

    pi5/          Raspberry Pi 5 acquisition and control
    analysis/     Visualization and the stimulus-decoding model
    firmware/     ESP32 / Arduino light-stimulus sketches
    sensors/      Standalone ambient-light sensor drivers
    experiments/  Earlier experiment runners (Arduino-protocol generation)

### `pi5/`

- **`ADS1263_pi5.py`** — driver for the ADS1263 32-bit ADC: register
  configuration (filter, data rate, PGA gain, chop mode), INTERFACE framing with
  status byte and checksum, and a software differential scan.
- **`config_pi5.py`** — GPIO/SPI backend. Uses libgpiod v2 with tolerance for
  naming differences across bindings, falling back to the v1 API and then to
  RPi.GPIO.
- **`mycotube_runner.py`** — the 50 ms experiment loop described above.
- **`plot_myco_run.py`** — offline analysis. Renders a 2×2 figure, one subplot
  per electrode channel, each overlaid with a computed light-intensity curve and
  vertical lap markers.

### `analysis/` — decoding the stimulus from the response

The instrument answers "what does the substrate do when we shine light on it?"
This code asks the inverse question: **given only the four electrode voltages,
can we recover what the light was doing?** If a model can, the recorded signal
carries real information about the stimulus rather than noise — which makes this
a validation of the whole measurement chain, not just a curiosity.

- **`train_brightness_fullrange_v4.py`** — two-stage model. A classifier
  predicts whether the light is on; a regressor predicts brightness (0–100),
  trained only on ON samples. Features are derived **exclusively** from the
  `diff*` columns; the `flux` (photodiode) and `panel_on` columns are
  deliberately excluded so the model cannot read the answer off a light sensor.
- **`train_brightness_fullrange_v2.py`** — the earlier linear version
  (logistic regression + ridge). Kept because the difference between the two is
  the point; see below.
- **`predict_brightness_fullrange_v4.py`** — inference. Smooths P(ON) with an
  EMA, converts it to a stable ON mask via hysteresis (separate on/off
  thresholds, so the state doesn't chatter at the boundary), then predicts
  brightness on ON samples.
- **`plot_myco_4diffs.py`** — 2×2 per-channel plot that shades the time regions
  where each electrode wall is lit, so stimulus and response line up visually.

**Why v4 exists.** v2 split the data chronologically across concatenated runs.
In a 20 Hz time series, adjacent samples are nearly identical, so a split that
lands mid-run puts near-duplicate rows on both sides and inflates the score.
v4 replaces it with **leave-one-run-out cross-validation** — hold out an entire
run, train on the others — which is the correct grouping for this data. v4 also
adds per-run drift removal (rolling median/MAD), lag features for the
substrate's response latency, and inverse-frequency bin weighting so the
regressor doesn't collapse toward the mid-range.

**Results, reported honestly.** On a held-out run the brightness regressor
reaches **MAE ≈ 9.2** on a 0–100 scale. The relevant comparison is a
predict-the-mean baseline, which scores MAE ≈ 24.3 on the same run — so the
model captures substantial real structure.

The ON/OFF presence metric looks near-perfect, but that number is not
meaningful: the light is on for ~98% of samples under this protocol, so a model
that always answers "on" scores ~98% too. Presence is trivial here; the
brightness regression is the actual task.

### `firmware/`

- **`esp32_led_controller/`** — the current stimulus firmware. Parses the `SET`
  / `STOP` protocol, renders vertical bars across the 256×64 surface, and runs
  the ON/OFF blink timing locally.
- **`esp32_hub75_brightness/`** — brightness-ramp sketch driving the same panel
  chain over a simpler `BRIGHT=NN` serial protocol.
- **`led_matrix_effects/`** — LED matrix effects sketch. Contains third-party
  code; see Attribution.

### `sensors/`

- **`tsl2591_pi.py`** — minimal TSL2591 ambient-light driver for the Pi over I²C
  (smbus2), with configurable integration time and gain.
- **`BH1750Driver.ino`** — self-contained BH1750 light-sensor driver for
  Arduino, with I²C address auto-detection and continuous/one-shot modes.

### `experiments/`

These target the earlier Arduino-protocol rig (`MASK` / `CLEAR` / `PING`).

- **`scan_and_drive_4sensors_rawlog.py`** — steps a non-cumulative panel
  sequence, logging raw differential voltages with no thresholding or filtering.
- **`calib.py`** — calibration: warm-up, baseline collection with median/MAD
  statistics, then eight directional flash segments (wall and electrode) to
  derive per-channel response sign and gain. Emits a JSON summary.
- **`mycopixelTestProgram.py`** — combined runner sampling the ADS1263, TSL2591
  (lux), and SHT31 (temperature/humidity) into a single CSV.

## Implementation notes

**First-sample discard on MUX change.** The ADS1263 converts one differential
pair at a time — there is no hardware multi-pair scan. After switching `INPMUX`,
the first conversion still reflects the previous input while the digital filter
settles. The driver programs the MUX, discards one conversion, and returns the
second. Without this, every channel reads like the first one.

**INPMUX write verification.** MUX writes are read back and compared, so a stuck
or mirrored channel surfaces as a warning instead of silently corrupting a run.

**Double-buffered rendering.** The ESP32 draws each frame into the back buffer
and flips atomically, which eliminated ghosting and scattered pixels seen when
drawing directly to the live buffer.

**Robust plot scaling.** Analysis clamps y-limits to the central 98th percentile,
so one transient doesn't flatten an entire trace.

**Monotonic timing.** The runner schedules ticks against `time.monotonic()` and
accumulates the target time rather than sleeping a fixed interval, so logging
doesn't drift over a long run.

## Setup

On the Raspberry Pi 5:

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python pi5/mycotube_runner.py --esp-port /dev/ttyUSB0

Flash `firmware/esp32_led_controller/` with the Arduino IDE (requires the
**ESP32-HUB75-MatrixPanel-I2S-DMA** and **ArduinoJson** libraries).

## Attribution

`ADS1263_pi5.py` began as Waveshare's reference driver for the ADS1263 and was
ported and hardened for the Raspberry Pi 5: migrated to libgpiod v2 with
fallbacks, explicit datasheet-aligned register and command definitions, INPMUX
write-verification, and the first-sample-discard differential scan described
above.

`firmware/led_matrix_effects/` adapts code from
[Aurora](https://github.com/pixelmatix/aurora) (© 2014 Jason Coon) and
LedEffects Plasma (© 2013 Robert Atkins), both MIT licensed. The original
copyright and permission notices are retained in the file.

Everything else — the experiment runner, the Pi↔microcontroller protocol, the
stimulus firmware, the calibration pipeline, and the analysis code — I wrote for
this project.

## Notes

Measurement data (run CSVs) and trained model artifacts (.joblib) are
excluded from version control. Hardware
integration on the lab Raspberry Pi 5 was handled by a teammate.
