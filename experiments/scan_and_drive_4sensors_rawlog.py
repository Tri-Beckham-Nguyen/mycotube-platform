#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
scan_and_drive_4sensors_rawlog.py

- Python tells Arduino which circles to show (MASK 0..15 or CLEAR).
- Python reads ADS1263 4 differential pairs (0..3) and logs raw voltages.
- NO detection, NO thresholds, NO filtering. Raw logging only.

Non-cumulative sequence:
  1 on, 1 off   => mask 0x01
  2 on, 2 off   => mask 0x02
  3 on, 3 off   => mask 0x04
  4 on, 4 off   => mask 0x08
  1,2,3,4 on; 1,2,3,4 off => mask 0x0F

CSV columns:
  t_s, phase(on/off), step_name, mask, V_pair1, V_pair2, V_pair3, V_pair4
"""

import time
import csv
import argparse
import serial

import ADS1263_pi5_new as ADS1263


def arduino_send_and_expect_ok(ser: serial.Serial, cmd: str, timeout_s: float = 2.0) -> str:
    """Send a line to Arduino and wait for OK/ERR."""
    ser.write((cmd.strip() + "\n").encode("utf-8"))
    ser.flush()

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.startswith("OK"):
            return line
        if line.startswith("ERR"):
            raise RuntimeError(f"Arduino error for '{cmd}': {line}")
        # ignore other noise
    raise TimeoutError(f"Arduino did not respond to '{cmd}' within {timeout_s}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Arduino serial port, e.g. /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--csv", default="scan_log.csv")

    # ADS1263 settings
    ap.add_argument("--rate", default="ADS1263_1200SPS", help="ADC1 rate key")
    ap.add_argument("--vref", type=float, default=2.5)

    # Timing
    ap.add_argument("--on-sec", type=float, default=5.0)
    ap.add_argument("--off-sec", type=float, default=5.0)
    ap.add_argument("--sample-hz", type=float, default=20.0)

    args = ap.parse_args()

    diff_indices = [0, 1, 2, 3]
    sample_dt = 1.0 / max(args.sample_hz, 1.0)

    # STRICTLY non-cumulative masks
    sequence = [
        ("1", 0x01),
        ("2", 0x02),
        ("3", 0x04),
        ("4", 0x08),
        ("1234", 0x0F),
    ]

    # ------------------- Open Arduino serial -------------------
    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(1.5)  # allow Arduino auto-reset on connect

    # Handshake / known state
    arduino_send_and_expect_ok(ser, "PING")
    arduino_send_and_expect_ok(ser, "CLEAR")

    # ------------------- Init ADC -------------------
    adc = ADS1263.ADS1263(vref=args.vref)

    if adc.ADS1263_init_ADC1(
        rate_key=args.rate,
        gain_key="ADS1263_GAIN_1",
        filter_key="FIR",
        chop_key="CHOP_ONLY",
    ) < 0:
        ser.close()
        adc.ADS1263_Exit()
        raise RuntimeError("ADS1263_init_ADC1 failed")

    adc.ADS1263_SetMode(1)  # differential
    _ = adc.ADS1263_GetAll(diff_indices)  # warm-up

    t0 = time.time()

    def log_for_duration(writer, step_name: str, phase: str, mask: int, duration_s: float):
        t_end = time.time() + duration_s
        while time.time() < t_end:
            elapsed = time.time() - t0

            codes = adc.ADS1263_GetAll(diff_indices)
            volts = [adc.code_to_volts(c) for c in codes]

            writer.writerow([
                f"{elapsed:.6f}",
                phase,
                step_name,
                int(mask),
                f"{volts[0]:+.9f}",
                f"{volts[1]:+.9f}",
                f"{volts[2]:+.9f}",
                f"{volts[3]:+.9f}",
            ])

            time.sleep(sample_dt)

    # ------------------- Run pattern + log -------------------
    try:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "phase", "step_name", "mask", "V_pair1", "V_pair2", "V_pair3", "V_pair4"])

            for (name, mask) in sequence:
                # ON
                arduino_send_and_expect_ok(ser, f"MASK {mask}")
                log_for_duration(w, name, "on", mask, args.on_sec)

                # OFF
                arduino_send_and_expect_ok(ser, "CLEAR")
                log_for_duration(w, name, "off", 0, args.off_sec)

        # leave panel off
        try:
            ser.write(b"CLEAR\n")
            ser.flush()
        except Exception:
            pass

    finally:
        ser.close()
        adc.ADS1263_Exit()

    print(f"Saved raw log to: {args.csv}")


if __name__ == "__main__":
    main()
