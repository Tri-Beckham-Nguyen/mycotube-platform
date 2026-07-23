#!/usr/bin/env python3
"""
train_brightness_fullrange_v2.py  (TRIMMED)

Two-stage training:
A) Presence model: light_on (brightness_pct > on_threshold) using diff-only features
B) Brightness regressor: predicts brightness_pct (0..100) using diff-only features,
   trained ONLY on ON samples.

Features include:
- rolling median of diff (level)
- rolling MAD of diff (robust variability)
- rolling median/mean of |d1| and |d2| (activity)

NEW:
- Trims the first --trim_s seconds of EACH run before training (default 75s).

Usage:
  python train_brightness_fullrange_v2.py --train_dir .\\Data\\Train --out brightness_fullrange_v2.joblib --trim_s 75
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def rolling_mad(x: pd.Series, win: int) -> pd.Series:
    med = x.rolling(win).median()
    return (x - med).abs().rolling(win).median()


def build_features(df: pd.DataFrame, diff_cols: list[str], dt: float, win: int) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    for c in diff_cols:
        x = df[c].astype(float)
        d1 = x.diff() / dt
        d2 = d1.diff() / dt

        ad1 = d1.abs()
        ad2 = d2.abs()

        # Level / state
        feat[f"{c}_med"] = x.rolling(win).median()
        feat[f"{c}_mad"] = rolling_mad(x, win)
        feat[f"{c}_mean"] = x.rolling(win).mean()

        # Activity / reactivity
        feat[f"{c}_ad1_med"] = ad1.rolling(win).median()
        feat[f"{c}_ad1_mean"] = ad1.rolling(win).mean()
        feat[f"{c}_ad2_med"] = ad2.rolling(win).median()
        feat[f"{c}_ad2_mean"] = ad2.rolling(win).mean()

    return feat


def load_rows(train_dir: str, win: int, on_threshold: float, trim_s: float, max_rows: int):
    files = sorted(glob(os.path.join(train_dir, "*.csv")))
    if not files:
        raise RuntimeError(f"No CSV files found in {train_dir}")

    X_all = []
    y_on_all = []
    y_bri_all = []

    used = 0
    skipped = 0

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            skipped += 1
            continue

        if "t_s" not in df.columns or "brightness_pct" not in df.columns:
            skipped += 1
            continue

        diff_cols = [c for c in df.columns if c.startswith("diff")]
        if not diff_cols:
            skipped += 1
            continue

        # ---- TRIM FIRST trim_s SECONDS ----
        try:
            df = df[df["t_s"].astype(float) >= float(trim_s)].copy()
        except Exception:
            skipped += 1
            continue

        if len(df) < 300:
            skipped += 1
            continue

        t = df["t_s"].to_numpy(float)
        if len(t) < 300:
            skipped += 1
            continue

        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            skipped += 1
            continue

        feat = build_features(df, diff_cols, dt, win)
        valid = feat.notna().all(axis=1)

        feat = feat.loc[valid]
        if len(feat) < 300:
            skipped += 1
            continue

        bri = df.loc[valid, "brightness_pct"].to_numpy(float)
        y_on = (bri > on_threshold).astype(int)

        X_all.append(feat.to_numpy(float))
        y_on_all.append(y_on)
        y_bri_all.append(bri)

        used += 1

    if not X_all:
        raise RuntimeError("No usable training data after trimming. Check trim_s and file durations.")

    X = np.vstack(X_all)
    y_on = np.concatenate(y_on_all)
    y_bri = np.concatenate(y_bri_all)

    # Subsample for speed
    if len(X) > max_rows:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X = X[idx]
        y_on = y_on[idx]
        y_bri = y_bri[idx]

    return X, y_on, y_bri, used, skipped


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--out", default="brightness_fullrange_v2.joblib")
    ap.add_argument("--roll_win", type=int, default=15)
    ap.add_argument("--on_threshold", type=float, default=1.0)
    ap.add_argument("--trim_s", type=float, default=75.0, help="Trim first N seconds of each run before training")
    ap.add_argument("--max_rows", type=int, default=300_000)
    ap.add_argument("--ridge_alpha", type=float, default=15.0)
    ap.add_argument("--clf_C", type=float, default=1.0)
    args = ap.parse_args()

    X, y_on, y_bri, used, skipped = load_rows(
        train_dir=args.train_dir,
        win=args.roll_win,
        on_threshold=args.on_threshold,
        trim_s=args.trim_s,
        max_rows=args.max_rows
    )

    split = int(0.8 * len(X))
    if split < 1000 or (len(X) - split) < 1000:
        raise RuntimeError("Not enough samples after trimming/feature-building. Add more runs or reduce trim.")

    Xtr, Xte = X[:split], X[split:]
    y_on_tr, y_on_te = y_on[:split], y_on[split:]
    y_bri_tr, y_bri_te = y_bri[:split], y_bri[split:]

    # A) Presence model
    presence_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1200, C=args.clf_C))
    ])
    presence_model.fit(Xtr, y_on_tr)
    y_on_hat = presence_model.predict(Xte)

    acc = accuracy_score(y_on_te, y_on_hat)
    prec = precision_score(y_on_te, y_on_hat, zero_division=0)
    rec = recall_score(y_on_te, y_on_hat, zero_division=0)
    f1 = f1_score(y_on_te, y_on_hat, zero_division=0)

    # B) Brightness regressor trained only on ON samples
    on_mask_tr = y_on_tr == 1
    on_mask_te = y_on_te == 1

    if int(np.sum(on_mask_tr)) < 1000 or int(np.sum(on_mask_te)) < 500:
        raise RuntimeError("Not enough ON samples after trimming. Lower threshold or add more data.")

    reg_model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=args.ridge_alpha))
    ])
    reg_model.fit(Xtr[on_mask_tr], y_bri_tr[on_mask_tr])

    y_bri_pred_on = reg_model.predict(Xte[on_mask_te])
    y_bri_pred_on = np.clip(y_bri_pred_on, 0.0, 100.0)

    mae_on = float(np.mean(np.abs(y_bri_te[on_mask_te] - y_bri_pred_on)))
    rmse_on = rmse(y_bri_te[on_mask_te], y_bri_pred_on)

    dump({
        "roll_win": args.roll_win,
        "on_threshold": args.on_threshold,
        "trim_s": args.trim_s,
        "presence_model": presence_model,
        "reg_model": reg_model,
        "note": "Two-stage: presence + ON-only brightness regression; trimmed first trim_s seconds per run"
    }, args.out)

    print("TRAINING COMPLETE (FULL RANGE V2, TRIMMED)")
    print(f"Used files    : {used}")
    print(f"Skipped files : {skipped}")
    print(f"Trimmed first : {args.trim_s:.1f} s per file")
    print(f"Saved model   : {args.out}")
    print("")
    print("Presence (ON/OFF):")
    print(f"  Accuracy  : {acc:.3f}")
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1        : {f1:.3f}")
    print("")
    print("Brightness (ON samples only):")
    print(f"  MAE (ON)   : {mae_on:.3f}")
    print(f"  RMSE (ON)  : {rmse_on:.3f}")


if __name__ == "__main__":
    main()
