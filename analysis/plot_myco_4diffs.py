#!/usr/bin/env python3
"""
plot_myco_4diffs.py — interactive 2x2 diffs + where the light is (no intensity)

Per CSV:
- 2x2 subplots: diff1..diff4
- Each subplot shows:
    - diff voltage (distinct color)
    - vertical markers showing WHEN the light is on that wall
    - lap markers: very light red vertical line at each new lap start

"Where the light is":
- Uses led_x1_256..led_x8_256 (spot centers along 0..256)
- A wall is considered "lit" at time t if ANY spot center is inside that wall interval:
    wall1: [0,64), wall2: [64,128), wall3: [128,192), wall4: [192,256)

Markers:
- When a wall is lit, we shade that time region lightly (easy to see).
- No right-axis intensity plot.

Run:
  python plot_myco_4diffs.py path\\to\\run.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


WALL_INTERVALS: Dict[int, Tuple[float, float]] = {
    1: (0.0, 64.0),
    2: (64.0, 128.0),
    3: (128.0, 192.0),
    4: (192.0, 256.0),
}

DIFF_COLORS = {1: "tab:blue", 2: "tab:green", 3: "tab:purple", 4: "tab:red"}


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "t_s" not in df.columns:
        raise ValueError(f"{path}: missing required column 't_s'")

    df["t_s"] = pd.to_numeric(df["t_s"], errors="coerce")
    df = df[np.isfinite(df["t_s"])].copy()
    df.sort_values("t_s", inplace=True)
    df.reset_index(drop=True, inplace=True)

    for i in range(1, 5):
        col = f"diff{i}_V"
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["led_lap", "led_n_spots"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for k in range(1, 9):
        col = f"led_x{k}_256"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def robust_ylim(
    y: np.ndarray,
    t: np.ndarray,
    ignore_first_s: float,
    lo_pct: float,
    hi_pct: float,
    pad_frac: float,
) -> Optional[Tuple[float, float]]:
    m = np.isfinite(y) & np.isfinite(t) & (t >= ignore_first_s)
    if np.count_nonzero(m) < 10:
        m = np.isfinite(y)
    yy = y[m]
    yy = yy[np.isfinite(yy)]
    if yy.size < 10:
        return None

    lo = np.percentile(yy, lo_pct)
    hi = np.percentile(yy, hi_pct)

    if lo == hi:
        span = abs(lo) if lo != 0 else 1.0
        lo -= 0.05 * span
        hi += 0.05 * span

    pad = pad_frac * (hi - lo)
    return (lo - pad, hi + pad)


def lap_start_times(t: np.ndarray, lap: Optional[np.ndarray]) -> np.ndarray:
    if lap is None or len(t) == 0 or len(lap) != len(t):
        return np.array([], dtype=float)
    lap_i = np.where(np.isfinite(lap), lap, 0.0)
    d = np.diff(lap_i)
    idx = np.where(d > 0)[0] + 1
    return t[idx]


def wall_lit_mask(df: pd.DataFrame, wall: int, max_spots: int = 8) -> np.ndarray:
    """
    Boolean mask: True when ANY spot center is inside the wall interval.
    Uses led_n_spots if present.
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=bool)

    w0, w1 = WALL_INTERVALS[wall]
    any_on = np.zeros(n, dtype=bool)
    n_spots = df["led_n_spots"].to_numpy(dtype=float) if "led_n_spots" in df.columns else None

    for k in range(1, max_spots + 1):
        col = f"led_x{k}_256"
        if col not in df.columns:
            continue
        xk = df[col].to_numpy(dtype=float)
        valid = np.isfinite(xk)
        if n_spots is not None:
            valid = valid & np.isfinite(n_spots) & (n_spots >= k)
        any_on |= valid & (xk >= w0) & (xk < w1)

    return any_on


def contiguous_true_intervals(t: np.ndarray, m: np.ndarray) -> List[Tuple[float, float]]:
    """
    Convert boolean mask m into a list of [t_start, t_end] intervals where m is True.
    """
    if len(t) == 0 or len(m) == 0:
        return []
    m = m.astype(bool)
    idx = np.where(m)[0]
    if idx.size == 0:
        return []

    intervals: List[Tuple[float, float]] = []
    start = idx[0]
    prev = idx[0]
    for j in idx[1:]:
        if j == prev + 1:
            prev = j
            continue
        intervals.append((t[start], t[prev]))
        start = j
        prev = j
    intervals.append((t[start], t[prev]))
    return intervals


def plot_one_file(csv_path: str, ignore_first_s: float) -> None:
    df = read_csv(csv_path)
    t = df["t_s"].to_numpy(dtype=float)
    lap = df["led_lap"].to_numpy(dtype=float) if "led_lap" in df.columns else None
    lap_times = lap_start_times(t, lap)

    diffs = [df[f"diff{i}_V"].to_numpy(dtype=float) for i in range(1, 5)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    axes = axes.ravel()

    fig_title = os.path.splitext(os.path.basename(csv_path))[0].replace("_", " ")
    fig.suptitle(fig_title)

    for i in range(4):
        wall = i + 1
        ax = axes[i]

        # Shade regions where THIS wall is lit
        lit = wall_lit_mask(df, wall=wall)
        intervals = contiguous_true_intervals(t, lit)
        for a, b in intervals:
            ax.axvspan(a, b, color="orange", alpha=0.10, zorder=0)

        # Lap start markers (very light red)
        for tt in lap_times:
            ax.axvline(tt, color="red", alpha=0.12, linewidth=1.0, zorder=1)

        # Diff trace
        ax.plot(t, diffs[i], color=DIFF_COLORS[wall], linewidth=1.0, label=f"diff{wall}_V", zorder=3)
        ax.set_ylabel("diff voltage (V)")
        ax.set_title(f"Wall {wall} (diff{wall})")
        ax.grid(True, alpha=0.3)

        yl = robust_ylim(diffs[i], t, ignore_first_s=ignore_first_s, lo_pct=1.0, hi_pct=99.0, pad_frac=0.10)
        if yl is not None:
            ax.set_ylim(*yl)

        # Keep legend simple
        ax.legend(loc="upper left")

    for ax in axes[2:]:
        ax.set_xlabel("time (s)")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="run CSV file(s)")
    ap.add_argument("--ignore-first-s", type=float, default=5.0, help="ignore first N seconds for diff y-limits")
    args = ap.parse_args()

    for p in args.csv:
        plot_one_file(p, ignore_first_s=args.ignore_first_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
