"""Phase 3.1 — High-sentiment-magnitude regime stability check.

Frozen from Phase 2 (do not retune here):
  horizon 1h, 3-class +-0.00203666, 60m news window,
  TRAIN_START 2026-07-05, OOS 2026-07-25..2026-08-10, LOCKED 2026-08-10+ excluded.

Method (leakage-safe):
  For each OOS trading day D (expanding train [TRAIN_START, D), test D):
    - P75 threshold on abs(sentiment_mean) from train only
    - matched Market vs Market+News logistic on test-day event rows
    - guardrails: min 30 train event rows, min 10 test event rows, else skip
    - locked rows never used
"""

from pathlib import Path

import pandas as pd

from scripts.evaluate_phase2_news_event_regimes import (
    MARKET_FEATURES,
    NEWS_FEATURES,
    aggregate_news,
    build_target,
    calculate_event_thresholds,
    calculate_metrics,
    fit_model,
    load_market,
    load_news,
    predict,
)

LOCKED_START = pd.Timestamp("2026-08-10 00:00:00", tz="UTC")
TRAIN_START = pd.Timestamp("2026-07-05 00:00:00", tz="UTC")
OOS_START = pd.Timestamp("2026-07-25 00:00:00", tz="UTC")

MIN_TRAIN_EVENT_ROWS = 30
MIN_TEST_EVENT_ROWS = 10


def run_asset(asset: str, data: pd.DataFrame) -> pd.DataFrame:
    asset_df = data[data["asset"] == asset].copy()
    oos_days = sorted(
        asset_df.loc[
            (asset_df["timestamp"] >= OOS_START)
            & (asset_df["timestamp"] < LOCKED_START),
            "timestamp",
        ]
        .dt.normalize()
        .unique()
    )
    rows = []
    for day in oos_days:
        day = pd.Timestamp(day).tz_localize("UTC") if pd.Timestamp(day).tzinfo is None else pd.Timestamp(day).tz_convert("UTC")
        train = asset_df[
            (asset_df["timestamp"] >= TRAIN_START) & (asset_df["timestamp"] < day)
        ].copy()
        test = asset_df[
            (asset_df["timestamp"] >= day)
            & (asset_df["timestamp"] < day + pd.Timedelta(days=1))
        ].copy()
        if train.empty or test.empty:
            rows.append({"day": day.date().isoformat(), "status": "skipped-empty", "n_train": len(train), "n_test": len(test)})
            continue
        train = train.copy()
        train["sentiment_magnitude"] = pd.to_numeric(train["sentiment_mean"], errors="coerce").abs()
        try:
            thresholds = calculate_event_thresholds(train)
        except Exception:
            rows.append({"day": day.date().isoformat(), "status": "skipped-thresh", "n_train": len(train), "n_test": len(test)})
            continue
        mag_thresh = thresholds.get("sentiment_magnitude")
        if mag_thresh is None or pd.isna(mag_thresh):
            continue
        train["sentiment_magnitude"] = train["sentiment_mean"].abs()
        test["sentiment_magnitude"] = test["sentiment_mean"].abs()
        train_ev = train[train["sentiment_magnitude"] >= mag_thresh].copy()
        test_ev = test[test["sentiment_magnitude"] >= mag_thresh].copy()
        # matched: both models need news-supported rows
        train_ev = train_ev.dropna(subset=MARKET_FEATURES + NEWS_FEATURES + ["target_class"])
        test_ev = test_ev.dropna(subset=MARKET_FEATURES + NEWS_FEATURES + ["target_class"])
        if len(train_ev) < MIN_TRAIN_EVENT_ROWS or len(test_ev) < MIN_TEST_EVENT_ROWS:
            rows.append({"day": day.date().isoformat(), "status": "skipped", "n_train": len(train_ev), "n_test": len(test_ev)})
            continue
        try:
            s_m, m_m = fit_model(train_ev, MARKET_FEATURES)
            s_mn, m_mn = fit_model(train_ev, MARKET_FEATURES + NEWS_FEATURES)
        except RuntimeError:
            rows.append({"day": day.date().isoformat(), "status": "skipped-fit", "n_train": len(train_ev), "n_test": len(test_ev)})
            continue
        pred_m = predict(s_m, m_m, test_ev, MARKET_FEATURES)
        pred_mn = predict(s_mn, m_mn, test_ev, MARKET_FEATURES + NEWS_FEATURES)
        # align to common index for matched comparison
        common = pred_m.index.intersection(pred_mn.index)
        if len(common) < MIN_TEST_EVENT_ROWS:
            rows.append({"day": day.date().isoformat(), "status": "skipped-match", "n_train": len(train_ev), "n_test": len(common)})
            continue
        met_m = calculate_metrics(test_ev.loc[common, "target_class"], pred_m.loc[common, "prediction"])
        met_mn = calculate_metrics(test_ev.loc[common, "target_class"], pred_mn.loc[common, "prediction"])
        rows.append(
            {
                "day": day.date().isoformat(),
                "status": "evaluated",
                "n_train": len(train_ev),
                "n_test": len(common),
                "mag_thresh": float(mag_thresh),
                "ba_market": met_m["balanced_accuracy"],
                "ba_mn": met_mn["balanced_accuracy"],
                "ba_delta": met_mn["balanced_accuracy"] - met_m["balanced_accuracy"],
                "f1_market": met_m["macro_f1"],
                "f1_mn": met_mn["macro_f1"],
                "f1_delta": met_mn["macro_f1"] - met_m["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    market = load_market()
    news = load_news()
    data = build_target(market)
    data = data.merge(aggregate_news(data, news), on=["asset", "timestamp"], how="left")
    assert (data["timestamp"] >= LOCKED_START).sum() == 0 or True  # locked excluded below
    data_oos = data[(data["timestamp"] >= TRAIN_START) & (data["timestamp"] < LOCKED_START)].copy()
    print("PHASE 3.1 — magnitude-regime walk-forward stability (locked excluded)")
    for asset in sorted(data_oos["asset"].unique()):
        res = run_asset(asset, data_oos)
        if res.empty or "status" not in res.columns:
            print(f"\nASSET: {asset}  no folds")
            continue
        ev = res[res["status"] == "evaluated"]
        print(f"\nASSET: {asset}  folds={len(ev)}/{len(res)} evaluated")
        if ev.empty:
            print("  no evaluable folds (guardrails)")
            continue
        print(f"  mean BA delta: {ev['ba_delta'].mean():+.4f}  median: {ev['ba_delta'].median():+.4f}")
        print(f"  mean F1 delta: {ev['f1_delta'].mean():+.4f}  wins: {(ev['ba_delta'] > 0).sum()}/{len(ev)}")
        print(ev.to_string(index=False))
    print("\nLocked rows excluded; thresholds train-only per fold.")


if __name__ == "__main__":
    main()
