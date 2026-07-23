#!/usr/bin/env python3
"""
predict_brightness_fullrange_v4.py

Loads brightness_fullrange_v4.joblib and a run CSV, then:
- trims first trim_s seconds
- builds same features (robust normalization + lags)
- predicts P(ON) with smoothing + hysteresis
- predicts brightness on ON samples using nonlinear regressor
- outputs actual vs predicted plot + metrics

Run:
  python predict_brightness_fullrange_v4.py
"""

import os
import numpy as np
import pandas as pd
from joblib import load


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    y[0] = float(x[0])
    for i in range(1, len(x)):
        y[i] = alpha * float(x[i]) + (1 - alpha) * y[i - 1]
    return y


def robust_center_scale(x: pd.Series, long_win: int) -> pd.Series:
    med = x.rolling(long_win, min_periods=max(5, long_win // 4)).median()
    mad = (x - med).abs().rolling(long_win, min_periods=max(5, long_win // 4)).median()
    mad = mad.clip(lower=1e-9)
    return (x - med) / mad


def build_features(df: pd.DataFrame, diff_cols: list[str], dt: float,
                   roll_win: int, drift_win_s: float,
                   lag_s_list: list[float]) -> pd.DataFrame:
    drift_win = max(int(round(drift_win_s / dt)), roll_win * 2)
    feat = pd.DataFrame(index=df.index)

    for c in diff_cols:
        x = df[c].astype(float)
        x_norm = robust_center_scale(x, drift_win)

        d1 = x_norm.diff() / dt
        d2 = d1.diff() / dt
        ad1 = d1.abs()
        ad2 = d2.abs()

        feat[f"{c}_x_med"] = x_norm.rolling(roll_win).median()
        feat[f"{c}_x_mean"] = x_norm.rolling(roll_win).mean()
        feat[f"{c}_ad1_med"] = ad1.rolling(roll_win).median()
        feat[f"{c}_ad1_mean"] = ad1.rolling(roll_win).mean()
        feat[f"{c}_ad2_med"] = ad2.rolling(roll_win).median()
        feat[f"{c}_ad2_mean"] = ad2.rolling(roll_win).mean()

    base_cols = list(feat.columns)
    for lag_s in lag_s_list:
        lag_n = int(round(lag_s / dt))
        if lag_n <= 0:
            continue
        for col in base_cols:
            feat[f"{col}_lag{lag_s:.1f}s"] = feat[col].shift(lag_n)

    return feat


def hysteresis_mask(p_on: np.ndarray, th_on: float, th_off: float) -> np.ndarray:
    """
    Turn p_on into a stable ON mask using hysteresis.
    """
    on = False
    out = np.zeros_like(p_on, dtype=bool)
    for i, p in enumerate(p_on):
        if not on and p >= th_on:
            on = True
        elif on and p <= th_off:
            on = False
        out[i] = on
    return out


def prompt_path(msg: str) -> str:
    while True:
        p = input(msg).strip().strip('"').strip("'")
        p = os.path.expanduser(p)
        if not p:
            print("Path cannot be empty.")
            continue
        if p.lower().endswith(".csv.csv"):
            p = p[:-4]
        if not os.path.exists(p):
            print(f"Not found: {p}")
            continue
        return p


def main():
    model_path = prompt_path("Enter model path (e.g., brightness_fullrange_v4.joblib): ")
    csv_path = prompt_path("Enter CSV path to predict: ")

    ema_alpha = float(input("EMA alpha for P(ON) [default=0.10]: ") or 0.10)
    th_on = float(input("Hysteresis ON threshold [default=0.60]: ") or 0.60)
    th_off = float(input("Hysteresis OFF threshold [default=0.40]: ") or 0.40)
    bri_smooth = float(input("EMA alpha for brightness smoothing [default=0.08]: ") or 0.08)

    bundle = load(model_path)
    trim_s = float(bundle["trim_s"])
    roll_win = int(bundle["roll_win"])
    drift_win_s = float(bundle["drift_win_s"])
    lag_s_list = [float(x) for x in bundle["lag_s_list"]]
    on_threshold = float(bundle["on_threshold"])
    presence = bundle["presence_model"]
    reg = bundle["reg_model"]

    df = pd.read_csv(csv_path)
    if "t_s" not in df.columns:
        raise ValueError("CSV must contain t_s.")
    diff_cols = [c for c in df.columns if c.startswith("diff")]
    if not diff_cols:
        raise ValueError("CSV must contain diff* columns.")

    df = df[df["t_s"].astype(float) >= trim_s].copy()
    if len(df) < 300:
        raise ValueError("Run too short after trim.")

    t = df["t_s"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid timebase.")

    feat = build_features(df, diff_cols, dt, roll_win, drift_win_s, lag_s_list)
    valid = feat.notna().all(axis=1)

    dfv = df.loc[valid].copy()
    X = feat.loc[valid].to_numpy(float)
    t2 = dfv["t_s"].to_numpy(float)

    p_on = presence.predict_proba(X)[:, 1]
    p_on = np.clip(p_on, 0.0, 1.0)
    p_on = ema(p_on, ema_alpha)

    on_mask = hysteresis_mask(p_on, th_on=th_on, th_off=th_off)

    bri_pred = np.zeros(len(t2), dtype=float)
    if np.any(on_mask):
        bri_raw = np.clip(reg.predict(X[on_mask]), 0.0, 100.0)
        bri_pred[on_mask] = bri_raw

    bri_pred = ema(bri_pred, bri_smooth)

    # Metrics if GT exists
    has_gt = "brightness_pct" in dfv.columns
    title = f"Brightness Prediction v4 (trim {trim_s:.0f}s)"
    if has_gt:
        y_true = dfv["brightness_pct"].to_numpy(float)
        mae = float(np.mean(np.abs(y_true - bri_pred)))
        r = rmse(y_true, bri_pred)
        title += f" | MAE={mae:.1f} RMSE={r:.1f}"
        print(f"\nMAE={mae:.3f} RMSE={r:.3f}")

        # presence accuracy relative to thresholded truth
        y_on_true = (y_true > on_threshold).astype(int)
        y_on_hat = on_mask.astype(int)
        pres_acc = float(np.mean(y_on_true == y_on_hat))
        print(f"Presence accuracy (this run) = {pres_acc:.3f}")

    out_prefix = os.path.splitext(csv_path)[0]
    out_csv = out_prefix + "_brightness_pred_v4.csv"
    out_png = out_prefix + "_brightness_pred_v4.png"

    out = pd.DataFrame({
        "t_s": t2,
        "p_on": p_on,
        "on_mask": on_mask.astype(int),
        "brightness_pred": bri_pred,
    })
    if has_gt:
        out["brightness_true"] = dfv["brightness_pct"].to_numpy(float)

    out.to_csv(out_csv, index=False)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    if has_gt:
        plt.plot(t2, out["brightness_true"].to_numpy(float), label="Actual brightness_pct")
    plt.plot(t2, bri_pred, label="Predicted brightness")
    plt.plot(t2, 100.0 * p_on, label="100 * P(ON)")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Brightness (%) / Probability scaled")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()
    plt.close()

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_png}\n")


if __name__ == "__main__":
    main()
