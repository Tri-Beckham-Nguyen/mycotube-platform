#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
fungal_calibration_ads1263.py

Calibration for fungal light-direction detection using ADS1263 on Raspberry Pi 5.

Differential mapping (ADC1):
    0 -> AIN0-1  (FRONT)
    1 -> AIN2-3  (RIGHT)
    2 -> AIN4-5  (BACK)
    3 -> AIN6-7  (LEFT)
    4 -> AIN8-9  (unused)

Pipeline:

1) Warm-up:
    - ADC scanning [0,1,2,3] for WARMUP seconds (data discarded).

2) Baseline (no deliberate light):
    - Collect BASELINE seconds of data.
    - Compute baseline median & MAD per channel.

3) Calibration flashes:
    - 4× "wall" flashes (front/right/back/left).
    - 4× "electrode" flashes (front/right/back/left).
    - Each segment FLASH seconds.
    - User is prompted and given a countdown.
    - All raw data saved as CSV.
    - Compute per-channel response sign and gain from the calibration.

Results:

- Saves:
    - baseline.csv
    - wall_front.csv, wall_right.csv, wall_back.csv, wall_left.csv
    - elec_front.csv, elec_right.csv, elec_back.csv, elec_left.csv
    - calibration_summary.json

The summary will be used by the detection script.
"""

import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import ADS1263_pi5_new as ADS1263  # New Pi 5-ready driver module


# ---------------------------------------------------------------------------
# Configuration & labels
# ---------------------------------------------------------------------------

DIFF_INDICES = [0, 1, 2, 3]  # FRONT, RIGHT, BACK, LEFT

PAIR_LABELS = {
    0: "AIN0-1 (FRONT)",
    1: "AIN2-3 (RIGHT)",
    2: "AIN4-5 (BACK)",
    3: "AIN6-7 (LEFT)",
    4: "AIN8-9",
}

WALL_CONDS = ["wall_front", "wall_right", "wall_back", "wall_left"]
ELEC_CONDS = ["elec_front", "elec_right", "elec_back", "elec_left"]
ALL_CONDS = WALL_CONDS + ELEC_CONDS

COND_HUMAN_TEXT = {
    "wall_front": "FLASH FRONT WALL (not directly on electrodes)",
    "wall_right": "FLASH RIGHT WALL (not directly on electrodes)",
    "wall_back": "FLASH BACK WALL (not directly on electrodes)",
    "wall_left": "FLASH LEFT WALL (not directly on electrodes)",

    "elec_front": "FLASH FRONT ELECTRODES",
    "elec_right": "FLASH RIGHT ELECTRODES",
    "elec_back": "FLASH BACK ELECTRODES",
    "elec_left": "FLASH LEFT ELECTRODES",
}

# Detection defaults to embed in the summary (can be overridden later)
FAST_WINDOW_SEC_DEFAULT = 5.0
BASELINE_TAU_SEC_DEFAULT = 120.0
BASELINE_UPDATE_Z_DEFAULT = 2.0
GLOBAL_Z_THRESHOLD_DEFAULT = 3.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def timestamp_dir_name(prefix: str = "calib_run") -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, times: List[float], values: List[List[float]]):
    """
    Write samples to CSV.

    times: list of elapsed seconds
    values: list of 4 lists [ch0_vals, ch1_vals, ch2_vals, ch3_vals],
            each same length as times.
    """
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "ch0_V", "ch1_V", "ch2_V", "ch3_V"])
        for k, t in enumerate(times):
            row = [t]
            for ch in range(4):
                row.append(values[ch][k])
            writer.writerow(row)


def collect_segment(
    adc: ADS1263.ADS1263,
    duration_s: float,
    diff_indices: List[int],
) -> Tuple[List[float], List[List[float]]]:
    """
    Collect a time segment of data for the given differential indices.

    Returns:
        times: list of elapsed seconds from segment start
        values: 4 lists (one per channel index in diff_indices order)
    """
    t0 = time.time()
    times: List[float] = []
    values: List[List[float]] = [[] for _ in diff_indices]

    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed >= duration_s:
            break

        codes = adc.ADS1263_GetAll(diff_indices)
        volts = [adc.code_to_volts(c) for c in codes]

        times.append(elapsed)
        for i, v in enumerate(volts):
            values[i].append(v)

    return times, values


def compute_baseline_stats(values: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given per-channel baseline values (4 lists), compute:
        - median per channel
        - MAD per channel (median absolute deviation)
    Returns arrays of shape (4,).
    """
    med = np.zeros(4, dtype=float)
    mad = np.zeros(4, dtype=float)

    for i in range(4):
        data = np.asarray(values[i], dtype=float)
        if data.size == 0:
            med[i] = 0.0
            mad[i] = 0.0
        else:
            m = np.median(data)
            med[i] = m
            mad[i] = np.median(np.abs(data - m))

    return med, mad


def compute_segment_medians(values: List[List[float]]) -> np.ndarray:
    """
    Compute median per channel for a calibration segment.
    """
    med = np.zeros(4, dtype=float)
    for i in range(4):
        data = np.asarray(values[i], dtype=float)
        med[i] = float(np.median(data)) if data.size > 0 else 0.0
    return med


def derive_signs_and_gains(
    baseline_med: np.ndarray,
    baseline_mad: np.ndarray,
    segment_meds: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    """
    Derive per-channel signs and gains from calibration data.

    Inputs:
        baseline_med: shape (4,)
        baseline_mad: shape (4,)
        segment_meds: dict cond -> medians array (4,)

    Returns:
        signs: array shape (4,), values +1 or -1
        gains: array shape (4,), >= 0
        deltas_by_cond: dict cond -> list of 4 delta medians (for summary)
    """
    # Delta medians per condition
    deltas_by_cond: Dict[str, List[float]] = {}
    for cond, med in segment_meds.items():
        deltas = (med - baseline_med).tolist()
        deltas_by_cond[cond] = deltas

    # Determine sign per channel: choose sign of the largest |delta| over all conditions
    signs = np.ones(4, dtype=float)
    for ch in range(4):
        best_mag = 0.0
        best_delta = 0.0
        for cond, deltas in deltas_by_cond.items():
            d = deltas[ch]
            mag = abs(d)
            if mag > best_mag:
                best_mag = mag
                best_delta = d
        if best_mag > 0.0:
            signs[ch] = 1.0 if best_delta >= 0 else -1.0
        else:
            signs[ch] = 1.0  # default

    # Determine gains per channel based on electrode flashes (primary responses)
    # Map sides to channel indices: FRONT=0, RIGHT=1, BACK=2, LEFT=3
    primary_deltas = []
    for side_idx, cond in enumerate(ELEC_CONDS):
        if cond in deltas_by_cond:
            d = deltas_by_cond[cond][side_idx]
            # Adjust sign so "positive" means excitation in preferred direction
            d_eff = signs[side_idx] * d
            primary_deltas.append(max(d_eff, 0.0))
        else:
            primary_deltas.append(0.0)

    primary_deltas = np.array(primary_deltas, dtype=float)
    nonzero = primary_deltas[primary_deltas > 0]
    if nonzero.size > 0:
        target = float(np.mean(nonzero))
    else:
        target = 1.0

    gains = np.ones(4, dtype=float)
    for side_idx in range(4):
        d_eff = primary_deltas[side_idx]
        if d_eff > 0:
            gains[side_idx] = target / d_eff
        else:
            gains[side_idx] = 1.0

    return signs, gains, deltas_by_cond


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fungal light-direction calibration using ADS1263 (4 differential channels)."
    )
    parser.add_argument(
        "--rate",
        default="ADS1263_1200SPS",
        help="ADS1263 ADC1 rate key (e.g. ADS1263_400SPS, ADS1263_100SPS, ADS1263_1200SPS)",
    )
    parser.add_argument(
        "--vref",
        type=float,
        default=2.5,
        help="Reference voltage used by the driver for code->volts (default: 2.5 V)",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=20.0,
        help="Warm-up duration in seconds (ADC running, data discarded).",
    )
    parser.add_argument(
        "--baseline-sec",
        type=float,
        default=180.0,
        help="Baseline duration in seconds (no light).",
    )
    parser.add_argument(
        "--flash-sec",
        type=float,
        default=10.0,
        help="Duration of each calibration flash segment (seconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store calibration data; default: calib_run_YYYYmmdd_HHMMSS",
    )
    args = parser.parse_args()

    out_dir_name = args.output_dir or timestamp_dir_name()
    out_dir = Path(out_dir_name)
    ensure_dir(out_dir)
    print(f"[INFO] Output directory: {out_dir}")

    # Initialize ADC
    adc = ADS1263.ADS1263(vref=args.vref)
    if adc.ADS1263_init_ADC1(
        rate_key=args.rate,
        gain_key="ADS1263_GAIN_1",
        filter_key="FIR",
        chop_key="CHOP_ONLY",
    ) < 0:
        print("[ERROR] ADS1263_init_ADC1 failed", file=sys.stderr)
        adc.ADS1263_Exit()
        sys.exit(1)

    adc.ADS1263_SetMode(1)  # differential mode

    print("[INFO] Driver module:", ADS1263.__file__)
    print(f"[INFO] vref = {adc.vref:.4f} V")
    for idx in DIFF_INDICES:
        print(f"[INFO] Using diff pair {idx}: {PAIR_LABELS[idx]}")

    # ---------------------- 1) Warm-up ----------------------
    if args.warmup_sec > 0:
        print(f"\n[PHASE] Warm-up: {args.warmup_sec:.1f} s (data discarded)")
        t_start = time.time()
        while time.time() - t_start < args.warmup_sec:
            _ = adc.ADS1263_GetAll(DIFF_INDICES)
            remaining = args.warmup_sec - (time.time() - t_start)
            remaining = max(0.0, remaining)
            print(f"  Warm-up remaining: {remaining:5.1f} s", end="\r")
        print("\n[PHASE] Warm-up complete.")

    # ---------------------- 2) Baseline ----------------------
    print(f"\n[PHASE] Baseline: {args.baseline_sec:.1f} s.")
    print("Please keep conditions stable and avoid deliberate light stimuli.")
    baseline_times, baseline_values = collect_segment(adc, args.baseline_sec, DIFF_INDICES)
    baseline_csv = out_dir / "baseline.csv"
    write_csv(baseline_csv, baseline_times, baseline_values)
    print(f"[INFO] Baseline data saved to {baseline_csv}")

    baseline_med, baseline_mad = compute_baseline_stats(baseline_values)
    print("[INFO] Baseline medians (V):", baseline_med)
    print("[INFO] Baseline MADs (V):   ", baseline_mad)

    # ---------------------- 3) Calibration segments ----------------------
    segment_meds: Dict[str, np.ndarray] = {}
    for cond in ALL_CONDS:
        label = COND_HUMAN_TEXT[cond]
        print(f"\n[PHASE] Calibration segment: {cond}")
        print(f"  Instruction: {label}")
        input("  Press Enter when ready...")
        for c in [3, 2, 1]:
            print(f"  Starting in {c}...")
            time.sleep(1.0)
        print(f"  Recording for {args.flash_sec:.1f} s...")

        seg_times, seg_values = collect_segment(adc, args.flash_sec, DIFF_INDICES)
        seg_csv = out_dir / f"{cond}.csv"
        write_csv(seg_csv, seg_times, seg_values)
        print(f"  [INFO] Segment data saved to {seg_csv}")

        med = compute_segment_medians(seg_values)
        segment_meds[cond] = med
        print(f"  [INFO] Segment medians (V): {med}")

    # ---------------------- 4) Compute signs & gains ----------------------
    signs, gains, deltas_by_cond = derive_signs_and_gains(
        baseline_med, baseline_mad, segment_meds
    )

    print("\n[CALIB] Derived signs per channel:", signs)
    print("[CALIB] Derived gains per channel:", gains)

    # Build summary
    summary = {
        "vref": adc.vref,
        "rate_key": args.rate,
        "baseline_sec": args.baseline_sec,
        "flash_sec": args.flash_sec,
        "baseline_median_V": baseline_med.tolist(),
        "baseline_mad_V": baseline_mad.tolist(),
        "channel_signs": signs.tolist(),
        "channel_gains": gains.tolist(),
        "deltas_by_condition": deltas_by_cond,
        "conditions": ALL_CONDS,
        "diff_indices": DIFF_INDICES,
        "pair_labels": {str(k): v for k, v in PAIR_LABELS.items()},
        "detection_params": {
            "FAST_WINDOW_SEC": FAST_WINDOW_SEC_DEFAULT,
            "BASELINE_TAU_SEC": BASELINE_TAU_SEC_DEFAULT,
            "BASELINE_UPDATE_Z": BASELINE_UPDATE_Z_DEFAULT,
            "GLOBAL_Z_THRESHOLD": GLOBAL_Z_THRESHOLD_DEFAULT,
        },
    }

    summary_path = out_dir / "calibration_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Calibration summary saved to {summary_path}")

    adc.ADS1263_Exit()
    print("\n[DONE] Calibration complete.")


if __name__ == "__main__":
    main()
