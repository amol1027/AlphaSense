"""Phase 4 pilot — next-hour range REGIME prediction (new question, not direction).

PRE-REGISTERED:
  Target: for bar at t, fut_range = (max(high) - min(low)) / close over bars
    in (t, t+1h]; label = 1 if fut_range > train median (train-only median
    recomputed per fold -> balanced by construction), else 0. Rows without a
    complete 4-bar future window are excluded.
  Features (all <= t): trailing range mean/max over past 4/16 bars,
    trailing return std over past 4/16 bars, volume change, high_low_range,
    close_open_return, plus normalized market set.
  Models: majority baseline vs LogisticRegression.
  Walk-forward: expanding train from 2024-01-01, monthly test, first test
    2024-07, locked 2026-08-10+ excluded; guardrails >=500 train / >=50 test.
  Locked: fit once on pre-locked, score locked once per asset.
  PASS per asset: >=8 folds AND median BA delta > +2% AND win rate >= 60%.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from src.features.market_features import add_normalized_market_features
from src.modeling.baseline import fit_majority_baseline

from scripts.evaluate_phase2_news_event_regimes import load_market

LOCKED_START = pd.Timestamp("2026-08-10 00:00:00", tz="UTC")
HISTORY_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
FIRST_TEST_MONTH = pd.Timestamp("2024-07-01 00:00:00", tz="UTC")

MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 50
REQUIRED_FOLDS = 8
MEDIAN_BAR = 0.02
WIN_BAR = 0.60


def add_future_range(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["asset", "exchange", "timestamp"]).reset_index(drop=True)
    g = df.groupby(["asset", "exchange"], sort=False)
    highs = pd.concat(
        [g["high"].shift(-k).rename(f"h{k}") for k in (1, 2, 3, 4)], axis=1
    )
    lows = pd.concat(
        [g["low"].shift(-k).rename(f"l{k}") for k in (1, 2, 3, 4)], axis=1
    )
    t4 = g["timestamp"].shift(-4)
    complete = t4.notna() & (t4 <= df["timestamp"] + pd.Timedelta(hours=1))
    df["fut_range"] = np.where(
        complete & highs.notna().all(axis=1) & lows.notna().all(axis=1),
        (highs.max(axis=1) - lows.min(axis=1)) / df["close"],
        np.nan,
    )
    return df


def add_trailing_vol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["asset", "exchange", "timestamp"]).reset_index(drop=True)
    g = df.groupby(["asset", "exchange"], sort=False)
    df["ret"] = g["close"].transform(lambda s: s / s.shift(1) - 1.0)
    for n, name in ((4, "1h"), (16, "4h")):
        df[f"range_mean_{name}"] = (
            g["high_low_range"].transform(lambda s, n=n: s.shift(1).rolling(n).mean())
        )
        df[f"range_max_{name}"] = (
            g["high_low_range"].transform(lambda s, n=n: s.shift(1).rolling(n).max())
        )
        df[f"ret_std_{name}"] = (
            g["ret"].transform(lambda s, n=n: s.shift(1).rolling(n).std())
        )
    return df


VOL_FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
    "range_mean_1h",
    "range_max_1h",
    "ret_std_1h",
    "range_mean_4h",
    "range_max_4h",
    "ret_std_4h",
]


def month_starts(first, last):
    cur = first
    while cur < last:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def run_asset(asset, data):
    ad = data[
        (data["asset"] == asset)
        & (data["timestamp"] >= HISTORY_START)
        & (data["timestamp"] < LOCKED_START)
    ].copy()
    rows = []
    for start in month_starts(FIRST_TEST_MONTH, LOCKED_START):
        end = start + pd.offsets.MonthBegin(1)
        train = ad[ad["timestamp"] < start].copy()
        test = ad[(ad["timestamp"] >= start) & (ad["timestamp"] < end)].copy()
        med = train["fut_range"].median()
        if pd.isna(med):
            continue
        train["label"] = (train["fut_range"] > med).astype(int)
        test["label"] = (test["fut_range"] > med).astype(int)
        train = train.dropna(subset=VOL_FEATURES + ["label"])
        test = test.dropna(subset=VOL_FEATURES + ["label"])
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        ytr, yte = train["label"].astype(int), test["label"].astype(int)
        pb = fit_majority_baseline(ytr).predict(test[VOL_FEATURES])
        sc = StandardScaler()
        m = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc.fit_transform(train[VOL_FEATURES]), ytr
        )
        pl = m.predict(sc.transform(test[VOL_FEATURES]))
        bb, bl = balanced_accuracy_score(yte, pb), balanced_accuracy_score(yte, pl)
        rows.append(
            {"month": start.date().isoformat(), "n_train": len(train),
             "n_test": len(test), "ba_base": bb, "ba_lr": bl, "ba_delta": bl - bb}
        )
    return pd.DataFrame(rows)


def verdict(folds):
    if len(folds) < REQUIRED_FOLDS:
        return f"FAIL (folds {len(folds)}/{REQUIRED_FOLDS})"
    med, win = float(folds["ba_delta"].median()), float((folds["ba_delta"] > 0).mean())
    if med > MEDIAN_BAR and win >= WIN_BAR:
        return f"PASS (med {med:+.3f}, win {win:.0%})"
    return f"FAIL (med {med:+.3f}, win {win:.0%})"


def main():
    data = load_market()
    data = add_normalized_market_features(data)
    data = add_trailing_vol(data)
    data = add_future_range(data)
    data = data.dropna(subset=["fut_range"]).reset_index(drop=True)
    print("PHASE 4 PILOT — high/low range-regime prediction (locked excluded from WF)")
    all_folds = []
    for asset in sorted(data["asset"].unique()):
        folds = run_asset(asset, data)
        folds["asset"] = asset
        all_folds.append(folds)
        print(f"\nASSET {asset}: {len(folds)} folds  {verdict(folds)}")
        if not folds.empty:
            print(
                f"  mean {folds['ba_delta'].mean():+.4f} median {folds['ba_delta'].median():+.4f} "
                f"wins {(folds['ba_delta'] > 0).sum()}/{len(folds)} "
                f"mean LR BA {folds.apply(lambda r: r['ba_base'] + r['ba_delta'], axis=1).mean():.4f}"
            )
        pre = data[(data["asset"] == asset) & (data["timestamp"] < LOCKED_START)].copy()
        lock = data[(data["asset"] == asset) & (data["timestamp"] >= LOCKED_START)].copy()
        med = pre["fut_range"].median()
        pre["label"] = (pre["fut_range"] > med).astype(int)
        lock["label"] = (lock["fut_range"] > med).astype(int)
        pre = pre.dropna(subset=VOL_FEATURES + ["label"])
        lock = lock.dropna(subset=VOL_FEATURES + ["label"])
        if pre.empty or lock.empty or lock["label"].nunique() < 2:
            print(f"  locked: n/a (pre={len(pre)}, locked={len(lock)})")
            continue
        ypr, ylo = pre["label"].astype(int), lock["label"].astype(int)
        pb = fit_majority_baseline(ypr).predict(lock[VOL_FEATURES])
        sc = StandardScaler()
        m = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc.fit_transform(pre[VOL_FEATURES]), ypr
        )
        pl = m.predict(sc.transform(lock[VOL_FEATURES]))
        print(
            f"  locked n={len(lock)}: base {balanced_accuracy_score(ylo, pb):.4f} "
            f"vs LR {balanced_accuracy_score(ylo, pl):.4f}"
        )
    pooled = pd.concat(all_folds, ignore_index=True)
    print(f"\nPOOLED: {len(pooled)} folds  {verdict(pooled)}")
    print("Done. Locked used once for final report only.")


if __name__ == "__main__":
    main()
