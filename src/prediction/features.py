"""Feature construction for serving. Mirrors the frozen validation builders.

Uses only bars at or before the prediction bar (shift(1) trailing stats).
"""

import pandas as pd

from src.features.market_features import add_normalized_market_features

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
]

REQUIRED_BARS = [
    "asset",
    "exchange",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

MIN_BARS = 6


def build_serving_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per input bar; last row is the prediction bar."""
    missing = set(REQUIRED_BARS) - set(bars.columns)
    if missing:
        raise ValueError(f"Missing bar columns: {sorted(missing)}")
    if len(bars) < MIN_BARS:
        raise ValueError(f"Need at least {MIN_BARS} bars, got {len(bars)}")
    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["asset"].nunique() > 1 or df["exchange"].nunique() > 1:
        raise ValueError("bars must cover a single asset/exchange")
    df = add_normalized_market_features(df)
    df["ret"] = df["close"] / df["close"].shift(1) - 1.0
    df["range_mean_1h"] = df["high_low_range"].shift(1).rolling(4).mean()
    df["range_max_1h"] = df["high_low_range"].shift(1).rolling(4).max()
    df["ret_std_1h"] = df["ret"].shift(1).rolling(4).std()
    return df
