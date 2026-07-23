#!/usr/bin/env python3
"""
plot_myco_run.py — plot a single MycoTube run CSV

One figure, 2x2 layout:
- 4 subplots: diff1..diff4 (each its own left axis)
- right axis in each subplot: effective light intensity (0..1) computed from ALL light spots
  relative to that subplot's diff wall (segment model)
- legend ONLY contains diff and light intensity
- light intensity line is orange
- lap vertical markers are shown but NOT in legend
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt


# diff wall centers (your mapping)
DIFF_CLOSE_X = {1: 32, 2: 96, 3: 160, 4: 224}

# wall geometry on the 256-perimeter
WALL_LEN = 64.0
WALL_HALF = WALL_LEN / 2.0

# proximity falloff distance (beyond the wall arc)
FALLOFF = 128.0


# -------------------- utilities --------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def circ_dist_256(a: float, b: float) -> float:
    """Shortest circular distance on [0,256)."""
    d = (a - b) % 256.0
    return min(d, 256.0 - d)


def circ_dist_to_arc_256(x: float, center: float, half_width: float) -> float:
    """
    Distance from x to a circular arc (interval) on [0,256).
    Arc is centered at `center` and spans length = 2*half_width.
    Returns 0 if x lies on the arc.
    """
    d_center = circ_dist_256(x, center)
    return max(0.0, d_center - half_width)


def robust_ylim(y: List[float], qlo: float = 0.01, qhi: float = 0.99) -> Tuple[float, float]:
    """Percentile-based y-limits with padding; ignores NaN/Inf."""
    y2 = [v for v in y if not (math.isnan(v) or math.isinf(v))]
    if not y2:
        return (-1.0, 1.0)

    ys = sorted(y2)
    n = len(ys)

    if n < 10:
        mn, mx = min(ys), max(ys)
        if mn == mx:
            return (mn - 1e-6, mx + 1e-6)
        pad = 0.05 * (mx - mn)
        return (mn - pad, mx + pad)

    lo_i = int(clamp(qlo * (n - 1), 0, n - 1))
    hi_i = int(clamp(qhi * (n - 1), 0, n - 1))
    lo, hi = ys[lo_i], ys[hi_i]

    if lo == hi:
        lo -= 1e-6
        hi += 1e-6

    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)


# -------------------- CSV loader --------------------

def load_csv_all_diffs(path: str, max_spots: int = 8):
    """
    Required columns:
      t_s
      diff1_V diff2_V diff3_V diff4_V

    Optional:
      led_lap
      led_x1_256 ... led_x{max_spots}_256
    """
    t: List[float] = []
    d1: List[float] = []
    d2: List[float] = []
    d3: List[float] = []
    d4: List[float] = []
    lap: List[int] = []
    xspots: List[List[float]] = []

    req = ["t_s", "diff1_V", "diff2_V", "diff3_V", "diff4_V"]

    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise RuntimeError("CSV has no header row.")

        for k in req:
            if k not in r.fieldnames:
                raise RuntimeError(f"CSV missing required column: {k}")

        for row in r:
            try:
                t_s = float(row["t_s"])
                v1 = float(row["diff1_V"])
                v2 = float(row["diff2_V"])
                v3 = float(row["diff3_V"])
                v4 = float(row["diff4_V"])

                lap_raw = row.get("led_lap", "")
                l = int(lap_raw) if lap_raw not in (None, "") else 0

                xs: List[float] = []
                for i in range(1, max_spots + 1):
                    k = f"led_x{i}_256"
                    v = row.get(k, "")
                    if v is None or v == "":
                        continue
                    xs.append(float(v))

                t.append(t_s)
                d1.append(v1)
                d2.append(v2)
                d3.append(v3)
                d4.append(v4)
                lap.append(l)
                xspots.append(xs)
            except Exception:
                continue

    if not t:
        raise RuntimeError("No valid data found in CSV.")

    return t, [d1, d2, d3, d4], lap, xspots


# -------------------- intensity computation --------------------

def effective_intensity_series(diff_idx: int, xspots: List[List[float]], max_spots: int) -> List[float]:
    """
    Per sample intensity (0..1):
      prox_k = clamp(1 - d_arc/FALLOFF, 0, 1)
      raw    = sum_k prox_k
      inten  = clamp(raw/max_spots, 0, 1)

    d_arc is distance to the wall ARC (length 64) around the wall center.
    """
    center = DIFF_CLOSE_X[diff_idx]
    denom = float(max_spots) if max_spots > 0 else 1.0

    out: List[float] = []
    for xs in xspots:
        s = 0.0
        for x in xs:
            d_arc = circ_dist_to_arc_256(x, center, WALL_HALF)
            prox = 1.0 - (d_arc / FALLOFF)
            if prox < 0.0:
                prox = 0.0
            elif prox > 1.0:
                prox = 1.0
            s += prox

        inten = s / denom
        if inten < 0.0:
            inten = 0.0
        elif inten > 1.0:
            inten = 1.0
        out.append(inten)

    return out


def lap_change_times(t: List[float], lap: List[int]) -> List[float]:
    if not lap:
        return []
    out: List[float] = []
    last = lap[0]
    for i in range(1, len(lap)):
        if lap[i] > last:
            out.append(t[i])
            last = lap[i]
    return out


# -------------------- plotting --------------------

def plot_run_2x2(csv_path: str, max_spots: int) -> None:
    t, diffs, lap, xspots = load_csv_all_diffs(csv_path, max_spots=max_spots)
    lap_times = lap_change_times(t, lap)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for idx in (1, 2, 3, 4):
        ax1 = axes_flat[idx - 1]
        dv = diffs[idx - 1]
        intensity = effective_intensity_series(idx, xspots, max_spots=max_spots)

        diff_line, = ax1.plot(
            t, dv,
            color="tab:blue",
            linewidth=1.2,
            label=f"diff{idx}"
        )
        ax1.grid(True)
        ax1.set_ylabel("diff V")

        ylo, yhi = robust_ylim(dv, 0.01, 0.99)
        ax1.set_ylim(ylo, yhi)

        for ts in lap_times:
            ax1.axvline(ts, color="red", linewidth=1.0, alpha=0.6)

        ax2 = ax1.twinx()
        inten_line, = ax2.plot(
            t, intensity,
            color="tab:orange",   # REQUIRED: orange
            linewidth=1.2,
            label="light intensity"
        )
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("I (0–1)")

        ax1.set_title(f"diff{idx}")

        # Legend: ONLY diff + intensity
        ax1.legend(handles=[diff_line, inten_line], loc="upper right")

    # shared x label on bottom row
    axes[1][0].set_xlabel("time (s)")
    axes[1][1].set_xlabel("time (s)")

    fig.suptitle("MycoTube run: diff1..diff4 with effective light intensity (multi-spot)")
    fig.tight_layout()
    plt.show()


# -------------------- CLI --------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to run CSV file")
    ap.add_argument("--max-spots", type=int, default=8, help="maximum led_x*_256 columns to read")
    args = ap.parse_args()

    plot_run_2x2(args.csv, max_spots=args.max_spots)


if __name__ == "__main__":
    main()