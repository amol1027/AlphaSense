"""Train frozen 1h volatility models per asset on pre-locked data.

Same builders/labels as validation (per-asset full-train median label).
Writes artifacts/volatility/{asset}.joblib (git-ignored, regenerable).
Locked data is never used here.
"""

from datetime import datetime, timezone

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.market_features import add_normalized_market_features
from src.prediction.artifacts import FEATURES, VolatilityBundle, save_bundle

from scripts.evaluate_phase2_news_event_regimes import load_market
from scripts.validate_phase4_volatility_pilot import (
    LOCKED_START,
    add_future_range,
    add_trailing_vol,
)


def main() -> None:
    data = load_market()
    data = add_normalized_market_features(data)
    data = add_trailing_vol(data)
    data = add_future_range(data)
    data = data.dropna(subset=["fut_range"]).reset_index(drop=True)
    trained_at = datetime.now(timezone.utc).isoformat()
    for asset in sorted(data["asset"].unique()):
        pre = data[
            (data["asset"] == asset) & (data["timestamp"] < LOCKED_START)
        ].copy()
        med = float(pre["fut_range"].median())
        pre["label"] = (pre["fut_range"] > med).astype(int)
        pre = pre.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
        y = pre["label"].astype(int)
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, random_state=42).fit(
            scaler.fit_transform(pre[FEATURES]), y
        )
        path = save_bundle(
            VolatilityBundle(
                asset=asset, features=list(FEATURES), scaler=scaler,
                model=model, range_median=med, n_train=len(pre),
                trained_at=trained_at,
            )
        )
        print(f"{asset}: n={len(pre)} median={med:.6f} -> {path}")


if __name__ == "__main__":
    main()
