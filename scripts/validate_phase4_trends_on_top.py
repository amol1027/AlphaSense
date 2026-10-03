"""Phase 4.6 — Google Trends attention on top of frozen volatility baseline.

PRE-REGISTERED:
  Attention (geo=IN, daily): rel_1d = interest[D-1] / median(past 30d),
    rel_7d = mean(rel over past 7d ending D-1). Relative form is robust to
    per-query 0-100 rescaling across fetch chunks. All inputs strictly before
    prediction day D (last completed day) — leakage-safe.
  Base: frozen 9 feats; target/labels/folds identical to pilot (monthly
    expanding WF from 2024-07, locked excluded, guardrails >=500/>=50).
  PASS: >=8 folds AND median BA delta (base+trends - base) > +2% AND win >= 60%.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
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

BASE_FEATS = [
    "return_15m", "return_30m", "return_1h",
    "high_low_range", "close_open_return", "volume_change",
    "range_mean_1h", "range_max_1h", "ret_std_1h",
]
TREND_FEATS = ["att_rel_1d", "att_rel_7d"]

TRENDS_PATH = "data/raw/attention/google_trends_daily.csv"
REQUIRED_FOLDS = 8
MEDIAN_BAR = 0.02
WIN_BAR = 0.60


def attach_attention(data: pd.DataFrame) -> pd.DataFrame:
    trends = pd.read_csv(TRENDS_PATH, parse_dates=["date"])
    trends["date"] = pd.to_datetime(trends["date"], utc=True).dt.normalize()
    data = data.copy()
    data["day"] = data["timestamp"].dt.normalize()
    out = []
    for asset, grp in data.groupby("asset"):
        t = trends[trends["asset"] == asset].sort_values("date")
        t = t.set_index("date")["interest"].astype(float)
        rel = t / t.rolling(30, min_periods=30).median().shift(1)
        grp = grp.copy()
        # rel[D] uses Trends data through D-1; lag by one more day so that a
        # prediction on day D only sees completed days through D-1.
        lag = rel.shift(1)
        grp["att_rel_1d"] = grp["day"].map(lag)
        grp["att_rel_7d"] = grp["day"].map(lag.rolling(7, min_periods=7).mean())
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def run_asset(asset, data):
    ad = data[
        (data["asset"] == asset)
        & (data["timestamp"] >= HISTORY_START)
        & (data["timestamp"] < LOCKED_START)
    ].copy()
    rows = []
    for start in month_starts(FIRST_TEST_MONTH, LOCKED_START):
        end = start + pd.offsets.MonthBegin(1)
        train = ad[ad["timestamp"] < start].dropna(
            subset=BASE_FEATS + TREND_FEATS + ["fut_range"]
        )
        test = ad[(ad["timestamp"] >= start) & (ad["timestamp"] < end)].dropna(
            subset=BASE_FEATS + TREND_FEATS + ["fut_range"]
        )
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        med = train["fut_range"].median()
        ytr = (train["fut_range"] > med).astype(int)
        yte = (test["fut_range"] > med).astype(int)
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        sc_b, sc_t = StandardScaler(), StandardScaler()
        m_b = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc_b.fit_transform(train[BASE_FEATS]), ytr
        )
        m_t = LogisticRegression(max_iter=1000, random_state=42).fit(
            sc_t.fit_transform(train[BASE_FEATS + TREND_FEATS]), ytr
        )
        bb = balanced_accuracy_score(yte, m_b.predict(sc_b.transform(test[BASE_FEATS])))
        bt = balanced_accuracy_score(
            yte, m_t.predict(sc_t.transform(test[BASE_FEATS + TREND_FEATS]))
        )
        rows.append({"month": start.date().isoformat(), "ba_base": bb,
                     "ba_tr": bt, "ba_delta": bt - bb})
    return pd.DataFrame(rows)


def main():
    data = load_market()
    data = add_normalized_market_features(data)
    data = add_trailing_vol(data)
    data = add_future_range(data)
    data = data.dropna(subset=["fut_range"]).reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = attach_attention(data)
    print("PHASE 4.6 — Trends attention on top of volatility baseline")
    all_folds = []
    for asset in sorted(data["asset"].unique()):
        folds = run_asset(asset, data)
        all_folds.append(folds.assign(asset=asset))
        if folds.empty:
            print(f"{asset}: no folds")
            continue
        med, win = float(folds["ba_delta"].median()), float((folds["ba_delta"] > 0).mean())
        v = "PASS" if (len(folds) >= REQUIRED_FOLDS and med > MEDIAN_BAR and win >= WIN_BAR) else "FAIL"
        print(f"{asset}: {len(folds)} folds med {med:+.4f} win {win:.0%} {v}")
    pooled = pd.concat(all_folds, ignore_index=True)
    med, win = float(pooled["ba_delta"].median()), float((pooled["ba_delta"] > 0).mean())
    v = "PASS" if (len(pooled) >= REQUIRED_FOLDS and med > MEDIAN_BAR and win >= WIN_BAR) else "FAIL"
    print(f"POOLED: {len(pooled)} folds med {med:+.4f} win {win:.0%} {v}")
    print("Done. Locked excluded.")


if __name__ == "__main__":
    main()
