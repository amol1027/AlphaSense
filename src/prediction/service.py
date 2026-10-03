"""Read-only prediction service over frozen per-asset bundles."""

from datetime import datetime, timezone

import pandas as pd

from src.prediction.artifacts import FEATURES, load_bundle
from src.prediction.features import build_serving_features

STALE_AFTER_DAYS = 7


def predict_range_regime(asset: str, bars: pd.DataFrame) -> dict:
    """Predict P(next-hour range above median) using only bars <= t.

    bars: 15m OHLCV history ending at prediction time t (last row = t).
    Returns JSON-serializable dict. Raises ValueError on bad input.
    """
    if "asset" not in bars.columns:
        raise ValueError("Missing bar columns: ['asset']")
    if set(bars["asset"].astype(str).str.upper()) != {asset.upper()}:
        raise ValueError(
            f"Requested asset {asset!r} does not match the supplied bars; "
            "bars must cover a single asset."
        )
    bundle = load_bundle(asset)
    feats = build_serving_features(bars)
    row = feats.iloc[[-1]]
    if row[FEATURES].isna().any(axis=None):
        raise ValueError(
            "Insufficient valid history: feature row contains NaN. "
            "Provide more trailing bars."
        )
    proba = float(bundle.model.predict_proba(bundle.scaler.transform(row[FEATURES]))[0, 1])
    prediction = int(proba >= 0.5)
    latest_price_change = float(
        (feats["close"].iloc[-1] / feats["close"].iloc[-2] - 1.0) * 100.0
    )
    ts = pd.Timestamp(row["timestamp"].iloc[0])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    age_days = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds() / 86400.0
    return {
        "asset": bundle.asset,
        "prediction_timestamp": ts.isoformat(),
        "target": "next-hour range above median",
        "probability_high_range": round(proba, 6),
        "prediction": prediction,
        "latest_price_change_pct": round(latest_price_change, 6),
        "range_median": bundle.range_median,
        "n_train": bundle.n_train,
        "data_age_days": round(age_days, 2),
        "stale": bool(age_days > STALE_AFTER_DAYS),
        "data_used": {
            "source": "Local 15-minute OHLCV bars",
            "bar_count": int(len(bars)),
            "features": {name: float(row[name].iloc[0]) for name in FEATURES},
            "feature_names": list(FEATURES),
            "training_rows": int(bundle.n_train),
            "trained_at": bundle.trained_at,
            "decision_threshold": 0.5,
        },
    }
