#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified experiment runner:
- ADS1263 (high-resolution analog)
- TSL2591 (lux)
- SHT31 (temperature, humidity)
- Single CSV export at end of experiment
"""

import time
import csv
from datetime import datetime

# ---- ADS1263 ----
import ADS1263_pi5_new as ADS1263

# ---- TSL2591 ----
from tsl2591_pi import TSL2591

# ---- SHT31 ----
import board
import busio
import adafruit_sht31d


# ============================================================
# Configuration
# ============================================================

EXPERIMENT_DURATION_S = 60          # total runtime
SAMPLE_PERIOD_S = 0.2               # sampling interval
CSV_FILENAME = "experiment_output.csv"

ADC_DIFF_CHANNELS = [0, 1, 2, 3]     # AIN0-1, 2-3, 4-5, 6-7
ADC_VREF = 5.0


# ============================================================
# Main experiment
# ============================================================

def run_experiment():

    # ---------------- ADC ----------------
    adc = ADS1263.ADS1263(vref=ADC_VREF)
    if adc.ADS1263_init_ADC1(rate_key='ADS1263_400SPS') < 0:
        raise RuntimeError("ADS1263 init failed")

    adc.ADS1263_SetMode(1)  # differential

    # ---------------- TSL2591 ----------------
    tsl = TSL2591(itime='100ms', gain='LOW')
    tsl.init()

    # ---------------- SHT31 ----------------
    i2c = busio.I2C(board.SCL, board.SDA)
    sht31 = adafruit_sht31d.SHT31D(i2c)

    # ---------------- Data buffer ----------------
    data_rows = []

    start_time = time.time()

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed >= EXPERIMENT_DURATION_S:
                break

            timestamp = datetime.utcnow().isoformat()

            # ---- ADS1263 ----
            adc_codes = adc.ADS1263_GetAll(ADC_DIFF_CHANNELS)
            adc_volts = [adc.code_to_volts(c) for c in adc_codes]

            # ---- TSL2591 ----
            lux, (c0, c1) = tsl.read_lux()

            # ---- SHT31 ----
            temperature_c = sht31.temperature
            humidity_rh = sht31.relative_humidity

            # ---- Record row ----
            row = [
                timestamp,
                elapsed,
                *adc_volts,
                lux,
                c0,
                c1,
                temperature_c,
                humidity_rh,
            ]
            data_rows.append(row)

            time.sleep(SAMPLE_PERIOD_S)

    finally:
        # Clean shutdown
        tsl.close()
        adc.ADS1263_Exit()

    # ---------------- CSV export ----------------
    write_csv(data_rows)


# ============================================================
# CSV Writer
# ============================================================

def write_csv(rows):

    header = [
        "timestamp_utc",
        "elapsed_s",
        "adc_ch0_1_V",
        "adc_ch2_3_V",
        "adc_ch4_5_V",
        "adc_ch6_7_V",
        "lux",
        "tsl_c0",
        "tsl_c1",
        "temperature_C",
        "humidity_percent",
    ]

    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[OK] Experiment complete — CSV written to: {CSV_FILENAME}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_experiment()

