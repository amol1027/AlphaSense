"""Phase 4.2 — Volatility-pilot feature ablation (find minimal set).

Same frozen target/eval as pilot (per-fold train-median label, expanding
monthly WF, locked scored once). Compares 7 feature groups; reports median
walk-forward BA and locked BA per group. No PASS/FAIL — descriptive.
"""

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
    add_future_range,
    add_trailing_vol,
    month_starts,
)

MARKET = [
    "return_15m", "return_30m", "return_1h",
    "high_low_range", "close_open_return", "volume_change",
]
TRAIL_1H = ["range_mean_1h", "range_max_1h", "ret_std_1h"]
TRAIL_4H = ["range_mean_4h", "range_max_4h", "ret_std_4h"]

GROUPS = {
    "full": MARKET + TRAIL_1H + TRAIL_4H,
    "market_only": MARKET,
    "trailing_only": TRAIL_1H + TRAIL_4H,
    "no_4h": MARKET + TRAIL_1H,
    "no_1h": MARKET + TRAIL_4H,
    "no_range": MARKET + ["ret_std_1h", "ret_std_4h"],
    "no_retstd": MARKET + ["range_mean_1h", "range_max_1h", "range_mean_4h", "range_max_4h"],
}


def run_group(asset, data, features):
    ad = data[
        (data["asset"] == asset)
        & (data["timestamp"] >= HISTORY_START)
        & (data["timestamp"] < LOCKED_START)
    ].copy()
    bas = []
    for start in month_starts(FIRST_TEST_MONTH, LOCKED_START):
        end = start + pd.offsets.MonthBegin(1)
        train = ad[ad["timestamp"] < start].copy()
        test = ad[(ad["timestamp"] >= start) & (ad["timestamp"] < end)].copy()
        med = train["fut_range"].median()
        if pd.isna(med):
            continue
        train["label"] = (train["fut_range"] > med).astype(int)
        test["label"] = (test["fut_range"] > med).astype(int)
        train = train.dropna(subset=features + ["label"])
        test = test.dropna(subset=features + ["label"])
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        ytr, yte = train["label"].astype(int), test["label"].astype(int)
        sc = StandardScaler()
        m = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc.fit_transform(train[features]), ytr
        )
        bas.append(balanced_accuracy_score(yte, m.predict(sc.transform(test[features]))))
    # locked
    pre = data[(data["asset"] == asset) & (data["timestamp"] < LOCKED_START)].copy()
    lock = data[(data["asset"] == asset) & (data["timestamp"] >= LOCKED_START)].copy()
    med = pre["fut_range"].median()
    pre["label"] = (pre["fut_range"] > med).astype(int)
    lock["label"] = (lock["fut_range"] > med).astype(int)
    pre = pre.dropna(subset=features + ["label"])
    lock = lock.dropna(subset=features + ["label"])
    locked_ba = float("nan")
    if not pre.empty and not lock.empty and lock["label"].nunique() == 2:
        ypr, ylo = pre["label"].astype(int), lock["label"].astype(int)
        sc = StandardScaler()
        m = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc.fit_transform(pre[features]), ypr
        )
        locked_ba = balanced_accuracy_score(ylo, m.predict(sc.transform(lock[features])))
    s = pd.Series(bas)
    return {"folds": len(s), "med_BA": s.median(), "mean_BA": s.mean(), "locked_BA": locked_ba}


def main():
    data = load_market()
    data = add_normalized_market_features(data)
    data = add_trailing_vol(data)
    data = add_future_range(data)
    data = data.dropna(subset=["fut_range"]).reset_index(drop=True)
    print("PHASE 4.2 — ablation (median walk-forward BA | locked BA)")
    assets = sorted(data["asset"].unique())
    summary = {}
    for gname, feats in GROUPS.items():
        row = {}
        for asset in assets:
            r = run_group(asset, data, feats)
            row[asset] = f"{r['med_BA']:.3f} | {r['locked_BA']:.3f} (n={r['folds']})"
            summary.setdefault(asset, {})[gname] = r
        print(f"\n{gname} ({len(feats)} feats):")
        for asset in assets:
            print(f"  {asset:10s} {row[asset]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
