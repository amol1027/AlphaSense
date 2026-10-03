"""Phase 4.3 — Volatility horizon sensitivity (frozen minimal 9-feat set).

Horizons: 30m (2 bars), 1h (4, reference), 2h (8), 4h (16). Target per
horizon: (max high - min low)/close over next n bars; label = above per-fold
train median; incomplete windows excluded. Same expanding monthly WF + locked
scoring as pilot. Descriptive: which horizons preserve the edge?
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from src.features.market_features import add_normalized_market_features
from src.modeling.baseline import fit_majority_baseline

from scripts.evaluate_phase2_news_event_regimes import load_market
from scripts.validate_phase4_volatility_pilot import (
    FIRST_TEST_MONTH,
    HISTORY_START,
    LOCKED_START,
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    add_trailing_vol,
    month_starts,
)

FEATS = [
    "return_15m", "return_30m", "return_1h",
    "high_low_range", "close_open_return", "volume_change",
    "range_mean_1h", "range_max_1h", "ret_std_1h",
]

HORIZONS = {"30m": 2, "1h": 4, "2h": 8, "4h": 16}


def add_future_range_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.sort_values(["asset", "exchange", "timestamp"]).reset_index(drop=True)
    g = df.groupby(["asset", "exchange"], sort=False)
    ks = list(range(1, n + 1))
    highs = pd.concat([g["high"].shift(-k).rename(f"h{k}") for k in ks], axis=1)
    lows = pd.concat([g["low"].shift(-k).rename(f"l{k}") for k in ks], axis=1)
    tn = g["timestamp"].shift(-n)
    complete = tn.notna() & (tn <= df["timestamp"] + pd.Timedelta(minutes=15 * n))
    df["fut_range"] = np.where(
        complete & highs.notna().all(axis=1) & lows.notna().all(axis=1),
        (highs.max(axis=1) - lows.min(axis=1)) / df["close"],
        np.nan,
    )
    return df


def run_asset(asset, data):
    ad = data[
        (data["asset"] == asset)
        & (data["timestamp"] >= HISTORY_START)
        & (data["timestamp"] < LOCKED_START)
    ].copy()
    out = []
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
        ytr, yte = train["label"].astype(int), test["label"].astype(int)
        pb = fit_majority_baseline(ytr).predict(test[FEATS])
        sc = StandardScaler()
        m = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc.fit_transform(train[FEATS]), ytr
        )
        pl = m.predict(sc.transform(test[FEATS]))
        bb, bl = balanced_accuracy_score(yte, pb), balanced_accuracy_score(yte, pl)
        out.append({"ba_delta": bl - bb, "ba_lr": bl})
    # locked
    pre = data[(data["asset"] == asset) & (data["timestamp"] < LOCKED_START)].copy()
    lock = data[(data["asset"] == asset) & (data["timestamp"] >= LOCKED_START)].copy()
    med = pre["fut_range"].median()
    pre["label"] = (pre["fut_range"] > med).astype(int)
    lock["label"] = (lock["fut_range"] > med).astype(int)
    pre = pre.dropna(subset=FEATS + ["label"])
    lock = lock.dropna(subset=FEATS + ["label"])
    locked = float("nan")
    if not pre.empty and not lock.empty and lock["label"].nunique() == 2:
        ypr, ylo = pre["label"].astype(int), lock["label"].astype(int)
        sc = StandardScaler()
        m = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc.fit_transform(pre[FEATS]), ypr
        )
        locked = balanced_accuracy_score(
            ylo, m.predict(sc.transform(lock[FEATS]))
        ) - balanced_accuracy_score(ylo, fit_majority_baseline(ypr).predict(lock[FEATS]))
    s = pd.DataFrame(out)
    if s.empty:
        return {"folds": 0, "med_delta": float("nan"), "win": float("nan"),
                "mean_lr": float("nan"), "locked_delta": locked}
    return {"folds": len(s), "med_delta": s["ba_delta"].median(),
            "win": (s["ba_delta"] > 0).mean(), "mean_lr": s["ba_lr"].mean(),
            "locked_delta": locked}


def main():
    base = load_market()
    base = add_normalized_market_features(base)
    base = add_trailing_vol(base)
    print("PHASE 4.3 — horizon sensitivity (median BA delta | win | locked delta)")
    assets = sorted(base["asset"].unique())
    for hname, n in HORIZONS.items():
        data = add_future_range_n(base.copy(), n)
        data = data.dropna(subset=["fut_range"]).reset_index(drop=True)
        print(f"\n{hname} ({n} bars):")
        for asset in assets:
            r = run_asset(asset, data)
            print(
                f"  {asset:10s} {r['med_delta']:+.3f} | {r['win']:.0%} "
                f"| locked {r['locked_delta']:+.3f} (n={r['folds']}, LR {r['mean_lr']:.3f})"
            )
    print("\nDone.")


if __name__ == "__main__":
    main()
