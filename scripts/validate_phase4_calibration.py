"""Phase 4.5 — Calibration of served volatility probabilities (measurement only).

Same frozen target/features/folds as pilot (monthly expanding WF from
2024-07, locked excluded). Collects out-of-fold predicted probabilities +
labels, then reports per-asset and pooled: Brier score vs constant
train-rate baseline, and decile reliability (mean predicted vs empirical
positive rate). No recalibration fitted here; no locked data used.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.market_features import add_normalized_market_features

from scripts.evaluate_phase2_news_event_regimes import load_market
from scripts.validate_phase4_volatility_pilot import (
    FIRST_TEST_MONTH,
    HISTORY_START,
    LOCKED_START,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    add_future_range,
    add_trailing_vol,
    month_starts,
)

FEATS = [
    "return_15m", "return_30m", "return_1h",
    "high_low_range", "close_open_return", "volume_change",
    "range_mean_1h", "range_max_1h", "ret_std_1h",
]

N_BINS = 10


def collect(asset, data):
    ad = data[
        (data["asset"] == asset)
        & (data["timestamp"] >= HISTORY_START)
        & (data["timestamp"] < LOCKED_START)
    ].copy()
    frames = []
    for start in month_starts(FIRST_TEST_MONTH, LOCKED_START):
        end = start + pd.offsets.MonthBegin(1)
        train = ad[ad["timestamp"] < start].copy()
        test = ad[(ad["timestamp"] >= start) & (ad["timestamp"] < end)].copy()
        med = train["fut_range"].median()
        if pd.isna(med):
            continue
        train["label"] = (train["fut_range"] > med).astype(int)
        test["label"] = (test["fut_range"] > med).astype(int)
        train = train.dropna(subset=FEATS + ["label"])
        test = test.dropna(subset=FEATS + ["label"])
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        ytr = train["label"].astype(int)
        base_rate = float(ytr.mean())
        sc = StandardScaler()
        m = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc.fit_transform(train[FEATS]), ytr
        )
        proba = m.predict_proba(sc.transform(test[FEATS]))[:, 1]
        frames.append(
            pd.DataFrame(
                {"y": test["label"].astype(int).to_numpy(),
                 "p": proba, "base": base_rate}
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def report(name, df):
    y, p = df["y"].to_numpy(), df["p"].to_numpy()
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((df["base"].to_numpy() - y) ** 2))
    print(f"\n{name}: n={len(df)} Brier model {brier:.4f} vs const {brier_base:.4f}")
    print("  bin | n | mean_p | emp_rate")
    order = np.argsort(p)
    for b in range(N_BINS):
        sl = slice(b * len(p) // N_BINS, (b + 1) * len(p) // N_BINS or None)
        idx = order[sl]
        if len(idx) == 0:
            continue
        print(
            f"  {b:3d} | {len(idx):5d} | {p[idx].mean():.3f} | {y[idx].mean():.3f}"
        )
    return brier, brier_base


def main():
    data = load_market()
    data = add_normalized_market_features(data)
    data = add_trailing_vol(data)
    data = add_future_range(data)
    data = data.dropna(subset=["fut_range"]).reset_index(drop=True)
    print("PHASE 4.5 — calibration (walk-forward out-of-fold, locked excluded)")
    pooled = []
    for asset in sorted(data["asset"].unique()):
        df = collect(asset, data)
        pooled.append(df)
        report(f"ASSET {asset}", df)
    report("POOLED", pd.concat(pooled, ignore_index=True))
    print("\nDone.")


if __name__ == "__main__":
    main()
