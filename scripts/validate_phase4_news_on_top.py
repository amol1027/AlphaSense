"""Phase 4.4 — News on top of frozen volatility baseline (the fair rematch).

PRE-REGISTERED:
  Target: 1h range-regime label (per-fold train median), frozen 9-feat base.
  News: 60m aggregation window (sentiment_mean/std, news_count,
    positive/negative_ratio), published_at <= t, matched rows only.
  Assets: RELIANCE + TCS (only ones with news coverage).
  Design: expanding cutoffs over OOS trading days; train = [2026-07-05, D),
    test = [D, LOCKED) pooled; guardrails >=30 train / >=20 test
    news-supported rows; locked excluded.
  PASS: >=8 folds AND median BA delta (base+news - base) > +2% AND win >= 60%.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from src.features.market_features import add_normalized_market_features
from src.modeling.baseline import fit_majority_baseline  # noqa: F401 (doc parity)

from scripts.evaluate_phase2_news_event_regimes import (
    NEWS_FEATURES,
    aggregate_news,
    load_market,
    load_news,
)
from scripts.validate_phase4_volatility_pilot import (
    LOCKED_START,
    add_future_range,
    add_trailing_vol,
)

TRAIN_START = pd.Timestamp("2026-07-05 00:00:00", tz="UTC")
OOS_START = pd.Timestamp("2026-07-25 00:00:00", tz="UTC")

BASE_FEATS = [
    "return_15m", "return_30m", "return_1h",
    "high_low_range", "close_open_return", "volume_change",
    "range_mean_1h", "range_max_1h", "ret_std_1h",
]

MIN_TRAIN = 30
MIN_TEST = 20
REQUIRED_FOLDS = 8
MEDIAN_BAR = 0.02
WIN_BAR = 0.60


def run_asset(asset, data):
    ad = data[data["asset"] == asset].copy()
    cutoffs = sorted(
        ad.loc[
            (ad["timestamp"] >= OOS_START) & (ad["timestamp"] < LOCKED_START),
            "timestamp",
        ]
        .dt.normalize()
        .unique()
    )
    rows = []
    for c in cutoffs:
        day = (
            pd.Timestamp(c).tz_localize("UTC")
            if pd.Timestamp(c).tzinfo is None
            else pd.Timestamp(c).tz_convert("UTC")
        )
        train = ad[(ad["timestamp"] >= TRAIN_START) & (ad["timestamp"] < day)].dropna(
            subset=BASE_FEATS + NEWS_FEATURES + ["fut_range"]
        )
        test = ad[(ad["timestamp"] >= day) & (ad["timestamp"] < LOCKED_START)].dropna(
            subset=BASE_FEATS + NEWS_FEATURES + ["fut_range"]
        )
        if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
            rows.append({"cutoff": day.date().isoformat(), "status": "skipped"})
            continue
        med = train["fut_range"].median()
        train["label"] = (train["fut_range"] > med).astype(int)
        test["label"] = (test["fut_range"] > med).astype(int)
        ytr, yte = train["label"].astype(int), test["label"].astype(int)
        sc_b, sc_n = StandardScaler(), StandardScaler()
        m_b = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc_b.fit_transform(train[BASE_FEATS]), ytr
        )
        m_n = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc_n.fit_transform(train[BASE_FEATS + NEWS_FEATURES]), ytr
        )
        bb = balanced_accuracy_score(yte, m_b.predict(sc_b.transform(test[BASE_FEATS])))
        bn = balanced_accuracy_score(
            yte, m_n.predict(sc_n.transform(test[BASE_FEATS + NEWS_FEATURES]))
        )
        rows.append(
            {"cutoff": day.date().isoformat(), "status": "evaluated",
             "n_train": len(train), "n_test": len(test),
             "ba_base": bb, "ba_news": bn, "ba_delta": bn - bb}
        )
    return pd.DataFrame(rows)


def main():
    market = load_market()
    news = load_news()
    data = add_normalized_market_features(market)
    data = add_trailing_vol(data)
    data = add_future_range(data)
    data = data.dropna(subset=["fut_range"]).reset_index(drop=True)
    data = data.merge(aggregate_news(data, news), on=["asset", "timestamp"], how="left")
    print("PHASE 4.4 — news on top of volatility baseline (locked excluded)")
    for asset in ("RELIANCE", "TCS"):
        res = run_asset(asset, data)
        ev = res[res["status"] == "evaluated"] if not res.empty else res
        print(f"\nASSET {asset}: {len(ev)}/{len(res)} folds")
        if ev.empty:
            print("  VERDICT: FAIL (no evaluable folds)")
            continue
        med, win = float(ev["ba_delta"].median()), float((ev["ba_delta"] > 0).mean())
        v = "PASS" if (len(ev) >= REQUIRED_FOLDS and med > MEDIAN_BAR and win >= WIN_BAR) else "FAIL"
        print(f"  median {med:+.4f} win {win:.0%} VERDICT: {v}")
        print(ev.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
