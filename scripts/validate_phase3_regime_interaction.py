"""Phase 3.3 — Market-regime x magnitude-regime interaction (last cheap test).

Frozen: same as 3.2 (1h, 3-class +-0.00203666, 60m window, OOS cutoffs, locked excluded).
Buckets (thresholds train-only per fold):
  VOL: high_vol = high_low_range >= train median; low_vol otherwise.
  TREND: high_move = abs(return_1h) >= train median; low_move otherwise.
Models trained once on all train magnitude-event rows; BA delta reported per test bucket.
Guardrails: >=30 train event rows, >=15 rows per test bucket, else skip bucket.
Pre-reg PASS per bucket: >=8 folds AND median BA delta > +2% AND win rate >= 60%.
"""

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
MIN_BUCKET_ROWS = 15
REQUIRED_FOLDS = 8
MEDIAN_BAR = 0.02
WIN_BAR = 0.60


def bucket_deltas(test_ev, pred_m, pred_mn, mask_hi, mask_lo):
    out = {}
    for name, mask in (("hi", mask_hi), ("lo", mask_lo)):
        idx = pred_m.index.intersection(pred_mn.index).intersection(
            test_ev.index[mask]
        )
        if len(idx) < MIN_BUCKET_ROWS:
            out[name] = None
            continue
        m = calculate_metrics(test_ev.loc[idx, "target_class"], pred_m.loc[idx, "prediction"])
        mn = calculate_metrics(test_ev.loc[idx, "target_class"], pred_mn.loc[idx, "prediction"])
        out[name] = {
            "n": len(idx),
            "delta": mn["balanced_accuracy"] - m["balanced_accuracy"],
        }
    return out


def run_asset(asset, data):
    ad = data[data["asset"] == asset].copy()
    cutoffs = sorted(
        ad.loc[
            (ad["timestamp"] >= OOS_START) & (ad["timestamp"] < LOCKED_START), "timestamp"
        ]
        .dt.normalize()
        .unique()
    )
    rows = []
    for c in cutoffs:
        day = pd.Timestamp(c).tz_localize("UTC") if pd.Timestamp(c).tzinfo is None else pd.Timestamp(c).tz_convert("UTC")
        train = ad[(ad["timestamp"] >= TRAIN_START) & (ad["timestamp"] < day)].copy()
        test = ad[(ad["timestamp"] >= day) & (ad["timestamp"] < LOCKED_START)].copy()
        if train.empty or test.empty:
            continue
        train["sentiment_magnitude"] = pd.to_numeric(train["sentiment_mean"], errors="coerce").abs()
        test["sentiment_magnitude"] = pd.to_numeric(test["sentiment_mean"], errors="coerce").abs()
        try:
            thresh = calculate_event_thresholds(train)["sentiment_magnitude"]
        except Exception:
            continue
        if pd.isna(thresh):
            continue
        vol_med = train["high_low_range"].median()
        mov_med = train["return_1h"].abs().median()
        if pd.isna(vol_med) or pd.isna(mov_med):
            continue
        train_ev = train[train["sentiment_magnitude"] >= thresh].dropna(
            subset=MARKET_FEATURES + NEWS_FEATURES + ["target_class"]
        )
        test_ev = test[test["sentiment_magnitude"] >= thresh].dropna(
            subset=MARKET_FEATURES + NEWS_FEATURES + ["target_class"]
        )
        if len(train_ev) < MIN_TRAIN_EVENT_ROWS or len(test_ev) < MIN_BUCKET_ROWS * 2:
            rows.append({"cutoff": day.date().isoformat(), "status": "skipped"})
            continue
        try:
            s_m, m_m = fit_model(train_ev, MARKET_FEATURES)
            s_mn, m_mn = fit_model(train_ev, MARKET_FEATURES + NEWS_FEATURES)
        except RuntimeError:
            rows.append({"cutoff": day.date().isoformat(), "status": "skipped-fit"})
            continue
        pred_m = predict(s_m, m_m, test_ev, MARKET_FEATURES)
        pred_mn = predict(s_mn, m_mn, test_ev, MARKET_FEATURES + NEWS_FEATURES)
        vol_hi = test_ev["high_low_range"] >= vol_med
        vol_lo = ~vol_hi
        mov_hi = test_ev["return_1h"].abs() >= mov_med
        mov_lo = ~mov_hi
        v = bucket_deltas(test_ev, pred_m, pred_mn, vol_hi, vol_lo)
        t = bucket_deltas(test_ev, pred_m, pred_mn, mov_hi, mov_lo)
        rows.append(
            {
                "cutoff": day.date().isoformat(),
                "status": "evaluated",
                "vol_hi": None if v["hi"] is None else v["hi"]["delta"],
                "vol_lo": None if v["lo"] is None else v["lo"]["delta"],
                "vol_hi_n": None if v["hi"] is None else v["hi"]["n"],
                "vol_lo_n": None if v["lo"] is None else v["lo"]["n"],
                "mov_hi": None if t["hi"] is None else t["hi"]["delta"],
                "mov_lo": None if t["lo"] is None else t["lo"]["delta"],
            }
        )
    return pd.DataFrame(rows)


def verdict(ev, col):
    s = ev.dropna(subset=[col])
    if len(s) < REQUIRED_FOLDS:
        return f"FAIL (folds {len(s)}/{REQUIRED_FOLDS})", float("nan"), float("nan")
    med = float(s[col].median())
    win = float((s[col] > 0).mean())
    v = "PASS" if (med > MEDIAN_BAR and win >= WIN_BAR) else "FAIL"
    return f"{v} (med {med:+.3f}, win {win:.0%})", med, win


def main():
    market = load_market()
    news = load_news()
    data = build_target(market)
    data = data.merge(aggregate_news(data, news), on=["asset", "timestamp"], how="left")
    data = data[(data["timestamp"] >= TRAIN_START) & (data["timestamp"] < LOCKED_START)].copy()
    print("PHASE 3.3 — regime interaction (locked excluded)")
    for asset in sorted(data["asset"].unique()):
        res = run_asset(asset, data)
        ev = res[res["status"] == "evaluated"] if not res.empty else res
        print(f"\nASSET {asset}: {len(ev)}/{len(res)} folds")
        if ev.empty:
            print("  no evaluable folds")
            continue
        for col in ("vol_hi", "vol_lo", "mov_hi", "mov_lo"):
            v, med, win = verdict(ev, col)
            print(f"  {col:7s}: {v}")
        print(ev.to_string(index=False))
    print("\nDone. Locked excluded.")


if __name__ == "__main__":
    main()
