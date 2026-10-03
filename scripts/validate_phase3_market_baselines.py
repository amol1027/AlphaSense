"""Phase 3.5 — Market-only 5-asset baselines (no news dependency).

PRE-REGISTERED:
  Target: binary next-hour direction (target_direction from 1h horizon).
  Features: normalized market set (return_15m/30m/1h, high_low_range,
    close_open_return, volume_change).
  Models: majority-class baseline (train labels only) vs LogisticRegression.
  Walk-forward: expanding train from 2024-01-01, monthly test periods,
    first test month 2024-07, last test window ends at LOCKED_START.
    Guardrails: >=500 train rows, >=50 test rows, both classes in train+test.
  Locked: [LOCKED_START, +infinity) never used in walk-forward; final report
    fits on all pre-locked data and scores locked once per asset.
  PASS per asset: >=8 folds AND median BA delta (logistic - majority) > +2%
    AND win rate >= 60%. Pooled verdict uses all-asset folds.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from src.features.horizon_targets import add_horizon_target
from src.features.market_features import add_normalized_market_features
from src.modeling.baseline import fit_majority_baseline

from scripts.evaluate_phase2_news_event_regimes import load_market

LOCKED_START = pd.Timestamp("2026-08-10 00:00:00", tz="UTC")
HISTORY_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
FIRST_TEST_MONTH = pd.Timestamp("2024-07-01 00:00:00", tz="UTC")
HORIZON = pd.Timedelta(hours=1)

MARKET_FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
]

MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 50
REQUIRED_FOLDS = 8
MEDIAN_BAR = 0.02
WIN_BAR = 0.60


def build_binary(data: pd.DataFrame) -> pd.DataFrame:
    df = add_normalized_market_features(data.copy())
    df = add_horizon_target(df, HORIZON)
    return df.dropna(subset=["target_direction"]).copy()


def month_starts(first: pd.Timestamp, last: pd.Timestamp):
    cur = first
    while cur < last:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def run_asset(asset: str, data: pd.DataFrame) -> pd.DataFrame:
    ad = data[
        (data["asset"] == asset)
        & (data["timestamp"] >= HISTORY_START)
        & (data["timestamp"] < LOCKED_START)
    ].copy()
    rows = []
    for start in month_starts(FIRST_TEST_MONTH, LOCKED_START):
        end = start + pd.offsets.MonthBegin(1)
        train = ad[ad["timestamp"] < start].dropna(
            subset=MARKET_FEATURES + ["target_direction"]
        )
        test = ad[(ad["timestamp"] >= start) & (ad["timestamp"] < end)].dropna(
            subset=MARKET_FEATURES + ["target_direction"]
        )
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        if train["target_direction"].nunique() < 2 or test["target_direction"].nunique() < 2:
            continue
        y_train = train["target_direction"].astype(int)
        y_test = test["target_direction"].astype(int)
        base = fit_majority_baseline(y_train)
        pred_base = base.predict(test[MARKET_FEATURES])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[MARKET_FEATURES])
        X_test = scaler.transform(test[MARKET_FEATURES])
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        pred_lr = model.predict(X_test)
        ba_b = balanced_accuracy_score(y_test, pred_base)
        ba_l = balanced_accuracy_score(y_test, pred_lr)
        rows.append(
            {
                "month": start.date().isoformat(),
                "n_train": len(train),
                "n_test": len(test),
                "acc_base": accuracy_score(y_test, pred_base),
                "acc_lr": accuracy_score(y_test, pred_lr),
                "ba_base": ba_b,
                "ba_lr": ba_l,
                "ba_delta": ba_l - ba_b,
            }
        )
    return pd.DataFrame(rows)


def verdict(folds: pd.DataFrame) -> str:
    if len(folds) < REQUIRED_FOLDS:
        return f"FAIL (folds {len(folds)}/{REQUIRED_FOLDS})"
    med = float(folds["ba_delta"].median())
    win = float((folds["ba_delta"] > 0).mean())
    if med > MEDIAN_BAR and win >= WIN_BAR:
        return f"PASS (med {med:+.3f}, win {win:.0%})"
    return f"FAIL (med {med:+.3f}, win {win:.0%})"


def main() -> None:
    data = build_binary(load_market())
    print("PHASE 3.5 — market-only 5-asset baselines (locked excluded from WF)")
    all_folds = []
    locked_rows = []
    for asset in sorted(data["asset"].unique()):
        folds = run_asset(asset, data)
        folds["asset"] = asset
        all_folds.append(folds)
        print(f"\nASSET {asset}: {len(folds)} folds  {verdict(folds)}")
        if not folds.empty:
            print(
                f"  mean BA delta {folds['ba_delta'].mean():+.4f}  "
                f"median {folds['ba_delta'].median():+.4f}  "
                f"wins {(folds['ba_delta'] > 0).sum()}/{len(folds)}"
            )
        # locked holdout, fit once on pre-locked
        pre = data[
            (data["asset"] == asset) & (data["timestamp"] < LOCKED_START)
        ].dropna(subset=MARKET_FEATURES + ["target_direction"])
        lock = data[
            (data["asset"] == asset) & (data["timestamp"] >= LOCKED_START)
        ].dropna(subset=MARKET_FEATURES + ["target_direction"])
        if pre.empty or lock.empty or lock["target_direction"].nunique() < 2:
            print(f"  locked: n/a (pre={len(pre)}, locked={len(lock)})")
            continue
        y_pre = pre["target_direction"].astype(int)
        y_lock = lock["target_direction"].astype(int)
        pb = fit_majority_baseline(y_pre).predict(lock[MARKET_FEATURES])
        sc = StandardScaler()
        m = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc.fit_transform(pre[MARKET_FEATURES]), y_pre
        )
        pl = m.predict(sc.transform(lock[MARKET_FEATURES]))
        locked_rows.append(
            {
                "asset": asset,
                "n_pre": len(pre),
                "n_locked": len(lock),
                "ba_base": balanced_accuracy_score(y_lock, pb),
                "ba_lr": balanced_accuracy_score(y_lock, pl),
                "ba_delta": balanced_accuracy_score(y_lock, pl)
                - balanced_accuracy_score(y_lock, pb),
            }
        )
        print(
            f"  locked n={len(lock)}: base {balanced_accuracy_score(y_lock, pb):.4f} "
            f"vs LR {balanced_accuracy_score(y_lock, pl):.4f}"
        )
    pooled = pd.concat(all_folds, ignore_index=True)
    print(f"\nPOOLED: {len(pooled)} folds  {verdict(pooled)}")
    print(pd.DataFrame(locked_rows).to_string(index=False))
    print("\nDone. Locked used once for final report only.")


if __name__ == "__main__":
    main()
