#!/usr/bin/env python3
"""
train_brightness_fullrange_v4.py

Full-range brightness model WITHOUT new data, using:
- trim first N seconds per file
- per-run robust normalization (median/MAD + drift removal)
- lag features (fungus latency)
- file-level CV (leave-one-run-out)
- nonlinear models:
    presence: HistGradientBoostingClassifier
    brightness: HistGradientBoostingRegressor (trained only on ON samples)
- sample weighting by brightness bins (reduces compression)

Usage:
  python train_brightness_fullrange_v4.py --train_dir .\Data\Train --out brightness_fullrange_v4.joblib
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def robust_center_scale(x: pd.Series, long_win: int) -> pd.Series:
    """
    Remove slow drift and normalize scale per-run:
    - subtract rolling median (long_win)
    - divide by rolling MAD (long_win) with floor
    """
    med = x.rolling(long_win, min_periods=max(5, long_win // 4)).median()
    mad = (x - med).abs().rolling(long_win, min_periods=max(5, long_win // 4)).median()
    mad = mad.clip(lower=1e-9)
    return (x - med) / mad


def build_features(df: pd.DataFrame, diff_cols: list[str], dt: float,
                   roll_win: int, drift_win_s: float,
                   lag_s_list: list[float]) -> pd.DataFrame:
    """
    Base features:
    - robust normalized signal x_norm
    - rolling median/mean of x_norm
    - rolling median/mean of |d1| and |d2| on x_norm

    Lag features:
    - shift base feature columns by lag samples
    """
    drift_win = max(int(round(drift_win_s / dt)), roll_win * 2)

    feat = pd.DataFrame(index=df.index)

    for c in diff_cols:
        x = df[c].astype(float)
        x_norm = robust_center_scale(x, drift_win)

        d1 = x_norm.diff() / dt
        d2 = d1.diff() / dt
        ad1 = d1.abs()
        ad2 = d2.abs()

        # Level/state
        feat[f"{c}_x_med"] = x_norm.rolling(roll_win).median()
        feat[f"{c}_x_mean"] = x_norm.rolling(roll_win).mean()

        # Activity
        feat[f"{c}_ad1_med"] = ad1.rolling(roll_win).median()
        feat[f"{c}_ad1_mean"] = ad1.rolling(roll_win).mean()
        feat[f"{c}_ad2_med"] = ad2.rolling(roll_win).median()
        feat[f"{c}_ad2_mean"] = ad2.rolling(roll_win).mean()

    # Add lagged copies
    base_cols = list(feat.columns)
    for lag_s in lag_s_list:
        lag_n = int(round(lag_s / dt))
        if lag_n <= 0:
            continue
        for col in base_cols:
            feat[f"{col}_lag{lag_s:.1f}s"] = feat[col].shift(lag_n)

    return feat


def brightness_bin_weights(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Weight samples inversely to brightness frequency to reduce mid-range bias.
    """
    y = np.asarray(y, dtype=float)
    bins = np.linspace(0, 100, n_bins + 1)
    idx = np.digitize(y, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins).astype(float)
    counts[counts == 0] = 1.0
    w = 1.0 / counts[idx]
    # Normalize weights to mean 1
    return w / np.mean(w)


def load_file_matrix(fp: str, trim_s: float, on_threshold: float,
                     roll_win: int, drift_win_s: float, lag_s_list: list[float]) -> dict | None:
    try:
        df = pd.read_csv(fp)
    except Exception:
        return None

    needed = {"t_s", "brightness_pct"}
    if not needed.issubset(df.columns):
        return None

    diff_cols = [c for c in df.columns if c.startswith("diff")]
    if not diff_cols:
        return None

    # Trim
    df = df[df["t_s"].astype(float) >= float(trim_s)].copy()
    if len(df) < 300:
        return None

    t = df["t_s"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None

    feat = build_features(df, diff_cols, dt, roll_win, drift_win_s, lag_s_list)
    valid = feat.notna().all(axis=1)

    feat = feat.loc[valid]
    if len(feat) < 300:
        return None

    X = feat.to_numpy(float)
    y_bri = df.loc[valid, "brightness_pct"].to_numpy(float)
    y_on = (y_bri > on_threshold).astype(int)

    return {
        "file": os.path.basename(fp),
        "dt": dt,
        "diff_cols": diff_cols,
        "X": X,
        "y_bri": y_bri,
        "y_on": y_on,
    }


def concat_except(items: list[dict], holdout_i: int):
    Xp, yon, yb = [], [], []
    for j, it in enumerate(items):
        if j == holdout_i:
            continue
        Xp.append(it["X"])
        yon.append(it["y_on"])
        yb.append(it["y_bri"])
    return np.vstack(Xp), np.concatenate(yon), np.concatenate(yb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--out", default="brightness_fullrange_v4.joblib")
    ap.add_argument("--trim_s", type=float, default=75.0)
    ap.add_argument("--on_threshold", type=float, default=1.0)
    ap.add_argument("--roll_win", type=int, default=15)
    ap.add_argument("--drift_win_s", type=float, default=20.0, help="Seconds for drift removal window")
    ap.add_argument("--lags", type=str, default="1,2,3,4,5", help="Lag seconds comma-separated")
    ap.add_argument("--max_files", type=int, default=10_000)
    args = ap.parse_args()

    lag_s_list = [float(x.strip()) for x in args.lags.split(",") if x.strip()]
    files = sorted(glob(os.path.join(args.train_dir, "*.csv")))[: args.max_files]
    if not files:
        raise RuntimeError(f"No CSV files in {args.train_dir}")

    items = []
    skipped = 0
    for fp in files:
        it = load_file_matrix(fp, args.trim_s, args.on_threshold, args.roll_win, args.drift_win_s, lag_s_list)
        if it is None:
            skipped += 1
            continue
        items.append(it)

    if len(items) < 3:
        raise RuntimeError("Need at least 3 usable runs for file-level CV.")

    # Models
    presence = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.08, max_iter=250, l2_regularization=0.1
    )

    reg = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.06, max_iter=400, l2_regularization=0.1,
        loss="squared_error"
    )

    # ---- Leave-one-file-out CV ----
    pres_acc = []
    bri_mae = []
    bri_rmse = []

    for i in range(len(items)):
        Xtr, y_on_tr, y_bri_tr = concat_except(items, i)
        Xte = items[i]["X"]
        y_on_te = items[i]["y_on"]
        y_bri_te = items[i]["y_bri"]

        presence.fit(Xtr, y_on_tr)
        p_on = presence.predict_proba(Xte)[:, 1]
        y_on_hat = (p_on >= 0.5).astype(int)
        pres_acc.append(float(np.mean(y_on_hat == y_on_te)))

        # Brightness: train only ON samples (train side)
        on_mask_tr = y_on_tr == 1
        on_mask_te = y_on_te == 1

        if np.sum(on_mask_tr) < 500 or np.sum(on_mask_te) < 200:
            # Not enough ON data in this fold; skip brightness metric for this fold
            continue

        w = brightness_bin_weights(y_bri_tr[on_mask_tr], n_bins=12)
        reg.fit(Xtr[on_mask_tr], y_bri_tr[on_mask_tr], sample_weight=w)

        y_pred_on = np.clip(reg.predict(Xte[on_mask_te]), 0.0, 100.0)
        y_true_on = y_bri_te[on_mask_te]

        bri_mae.append(float(np.mean(np.abs(y_true_on - y_pred_on))))
        bri_rmse.append(rmse(y_true_on, y_pred_on))

    print("CROSS-VALIDATION (leave-one-run-out)")
    print(f"Usable runs   : {len(items)}")
    print(f"Skipped runs  : {skipped}")
    print(f"Trim first    : {args.trim_s:.1f} s")
    print(f"Lags (s)      : {lag_s_list}")
    print("")
    print(f"Presence accuracy (avg): {np.mean(pres_acc):.3f}")

    if bri_mae:
        print(f"Brightness MAE on ON (avg): {np.mean(bri_mae):.2f}")
        print(f"Brightness RMSE on ON (avg): {np.mean(bri_rmse):.2f}")
    else:
        print("Brightness CV metrics: insufficient ON samples in folds.")

    # ---- Fit final models on ALL data ----
    X_all = np.vstack([it["X"] for it in items])
    y_on_all = np.concatenate([it["y_on"] for it in items])
    y_bri_all = np.concatenate([it["y_bri"] for it in items])

    presence.fit(X_all, y_on_all)

    on_mask_all = y_on_all == 1
    if np.sum(on_mask_all) < 1000:
        raise RuntimeError("Not enough ON samples overall after trimming.")
    w_all = brightness_bin_weights(y_bri_all[on_mask_all], n_bins=12)
    reg.fit(X_all[on_mask_all], y_bri_all[on_mask_all], sample_weight=w_all)

    bundle = {
        "version": "v4",
        "trim_s": args.trim_s,
        "on_threshold": args.on_threshold,
        "roll_win": args.roll_win,
        "drift_win_s": args.drift_win_s,
        "lag_s_list": lag_s_list,
        "presence_model": presence,
        "reg_model": reg,
        "note": "v4: trim + robust per-run normalization + lag features + HGB models + bin weighting + file-level CV"
    }
    dump(bundle, args.out)

    print("")
    print("FINAL TRAINING COMPLETE")
    print(f"Saved model: {args.out}")


if __name__ == "__main__":
    main()
