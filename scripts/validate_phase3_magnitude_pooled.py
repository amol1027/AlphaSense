"""Phase 3.2 — Expanding pooled validation of high-sentiment-magnitude regime.

PRE-REGISTERED (frozen before seeing results):
  Frozen config: 1h horizon, 3-class +-0.00203666, 60m news window,
    TRAIN_START 2026-07-05, OOS 2026-07-25..2026-08-10, LOCKED 2026-08-10+ excluded.
  Design: for each OOS trading-day cutoff D:
    train = [TRAIN_START, D), test = [D, LOCKED)  (pooled future, matched rows)
    P75 abs(sentiment_mean) threshold from train only.
  Guardrails: >=30 train event rows, >=20 pooled test rows, else skip fold.
  Bootstrap: B=1000 resamples of matched test rows -> 95% CI on BA delta, P(delta>0).
  PASS per asset requires ALL of:
    (a) >=8 evaluable folds,
    (b) median BA delta > +0.02,
    (c) >=60% folds with BA delta > 0.
  Anything else = FAIL (regime not stable enough to carry forward).

Leakage: thresholds train-only, locked never used.
"""

import numpy as np
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
MIN_TEST_EVENT_ROWS = 20
N_BOOT = 1000
SEED = 42

# Pre-registered pass/fail
REQUIRED_FOLDS = 8
MEDIAN_BA_DELTA_BAR = 0.02
WIN_RATE_BAR = 0.60


def bootstrap_ba_delta(
    actual: pd.Series, pred_m: pd.Series, pred_mn: pd.Series, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.asarray(actual.index)
    # paired bootstrap on matched test rows using macro-averaged recall as BA proxy
    # (balanced_accuracy on resample); fast vectorized loop over B=1000, n small
    from sklearn.metrics import balanced_accuracy_score

    deltas = np.empty(N_BOOT)
    a = actual.to_numpy()
    pm = pred_m.to_numpy()
    pn = pred_mn.to_numpy()
    n = len(a)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        try:
            deltas[b] = balanced_accuracy_score(
                a[s], pn[s]
            ) - balanced_accuracy_score(a[s], pm[s])
        except Exception:
            deltas[b] = np.nan
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return {"lo": float("nan"), "hi": float("nan"), "p_pos": float("nan")}
    return {
        "lo": float(np.quantile(deltas, 0.025)),
        "hi": float(np.quantile(deltas, 0.975)),
        "p_pos": float((deltas > 0).mean()),
    }


def run_asset(asset: str, data: pd.DataFrame) -> pd.DataFrame:
    asset_df = data[data["asset"] == asset].copy()
    cutoffs = sorted(
        asset_df.loc[
            (asset_df["timestamp"] >= OOS_START)
            & (asset_df["timestamp"] < LOCKED_START),
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
        train = asset_df[
            (asset_df["timestamp"] >= TRAIN_START) & (asset_df["timestamp"] < day)
        ].copy()
        test = asset_df[
            (asset_df["timestamp"] >= day) & (asset_df["timestamp"] < LOCKED_START)
        ].copy()
        if train.empty or test.empty:
            rows.append({"cutoff": day.date().isoformat(), "status": "skipped-empty"})
            continue
        train["sentiment_magnitude"] = pd.to_numeric(
            train["sentiment_mean"], errors="coerce"
        ).abs()
        test["sentiment_magnitude"] = pd.to_numeric(
            test["sentiment_mean"], errors="coerce"
        ).abs()
        try:
            thresh = calculate_event_thresholds(train)["sentiment_magnitude"]
        except Exception:
            rows.append({"cutoff": day.date().isoformat(), "status": "skipped-thresh"})
            continue
        if pd.isna(thresh):
            rows.append({"cutoff": day.date().isoformat(), "status": "skipped-thresh"})
            continue
        train_ev = train[train["sentiment_magnitude"] >= thresh].dropna(
            subset=MARKET_FEATURES + NEWS_FEATURES + ["target_class"]
        )
        test_ev = test[test["sentiment_magnitude"] >= thresh].dropna(
            subset=MARKET_FEATURES + NEWS_FEATURES + ["target_class"]
        )
        if len(train_ev) < MIN_TRAIN_EVENT_ROWS or len(test_ev) < MIN_TEST_EVENT_ROWS:
            rows.append(
                {
                    "cutoff": day.date().isoformat(),
                    "status": "skipped-guardrail",
                    "n_train": len(train_ev),
                    "n_test": len(test_ev),
                }
            )
            continue
        try:
            s_m, m_m = fit_model(train_ev, MARKET_FEATURES)
            s_mn, m_mn = fit_model(train_ev, MARKET_FEATURES + NEWS_FEATURES)
        except RuntimeError:
            rows.append({"cutoff": day.date().isoformat(), "status": "skipped-fit"})
            continue
        pred_m = predict(s_m, m_m, test_ev, MARKET_FEATURES)
        pred_mn = predict(s_mn, m_mn, test_ev, MARKET_FEATURES + NEWS_FEATURES)
        common = pred_m.index.intersection(pred_mn.index)
        if len(common) < MIN_TEST_EVENT_ROWS:
            rows.append({"cutoff": day.date().isoformat(), "status": "skipped-match"})
            continue
        met_m = calculate_metrics(
            test_ev.loc[common, "target_class"], pred_m.loc[common, "prediction"]
        )
        met_mn = calculate_metrics(
            test_ev.loc[common, "target_class"], pred_mn.loc[common, "prediction"]
        )
        boot = bootstrap_ba_delta(
            test_ev.loc[common, "target_class"],
            pred_m.loc[common, "prediction"],
            pred_mn.loc[common, "prediction"],
            SEED + len(rows),
        )
        rows.append(
            {
                "cutoff": day.date().isoformat(),
                "status": "evaluated",
                "n_train": len(train_ev),
                "n_test": len(common),
                "mag_thresh": float(thresh),
                "ba_market": met_m["balanced_accuracy"],
                "ba_mn": met_mn["balanced_accuracy"],
                "ba_delta": met_mn["balanced_accuracy"] - met_m["balanced_accuracy"],
                "ci_lo": boot["lo"],
                "ci_hi": boot["hi"],
                "p_pos": boot["p_pos"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    market = load_market()
    news = load_news()
    data = build_target(market)
    data = data.merge(aggregate_news(data, news), on=["asset", "timestamp"], how="left")
    data_oos = data[
        (data["timestamp"] >= TRAIN_START) & (data["timestamp"] < LOCKED_START)
    ].copy()
    print("PHASE 3.2 — expanding pooled folds + bootstrap (locked excluded)")
    print(
        f"PASS requires: >={REQUIRED_FOLDS} folds, median BA delta > +{MEDIAN_BA_DELTA_BAR:.0%}, "
        f"win rate >= {WIN_RATE_BAR:.0%}"
    )
    for asset in sorted(data_oos["asset"].unique()):
        res = run_asset(asset, data_oos)
        ev = res[res["status"] == "evaluated"].copy() if not res.empty else res
        print(f"\nASSET: {asset} folds={len(ev)}/{len(res)}")
        if ev.empty:
            print("  VERDICT: FAIL (no evaluable folds)")
            continue
        med = float(ev["ba_delta"].median())
        win = float((ev["ba_delta"] > 0).mean())
        verdict = (
            "PASS"
            if (len(ev) >= REQUIRED_FOLDS and med > MEDIAN_BA_DELTA_BAR and win >= WIN_RATE_BAR)
            else "FAIL"
        )
        print(f"  median BA delta: {med:+.4f}  win rate: {win:.0%}  VERDICT: {verdict}")
        print(
            ev[
                ["cutoff", "n_train", "n_test", "ba_market", "ba_mn", "ba_delta", "ci_lo", "ci_hi", "p_pos"]
            ].to_string(index=False)
        )
    print("\nLocked rows excluded; thresholds train-only per fold.")


if __name__ == "__main__":
    main()
