import pandas as pd


NORMALIZED_MARKET_FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
]


def add_normalized_market_features(
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add scale-independent market features.

    The input must contain:
        asset
        exchange
        timestamp
        open
        high
        low
        close
        volume

    All rolling/lagged features use only historical
    observations. No future information is used.
    """

    required_columns = {
        "asset",
        "exchange",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required_columns
        - set(market_df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required market columns: "
            f"{sorted(missing)}"
        )

    df = market_df.copy()

    df = df.sort_values(
        [
            "asset",
            "exchange",
            "timestamp",
        ]
    ).reset_index(drop=True)

    group_columns = [
        "asset",
        "exchange",
    ]

    grouped_close = df.groupby(
        group_columns,
        sort=False,
    )["close"]

    grouped_volume = df.groupby(
        group_columns,
        sort=False,
    )["volume"]

    # 15-minute return.
    df["return_15m"] = (
        df["close"]
        / grouped_close.shift(1)
        - 1.0
    )

    # 30-minute return.
    df["return_30m"] = (
        df["close"]
        / grouped_close.shift(2)
        - 1.0
    )

    # 1-hour return.
    df["return_1h"] = (
        df["close"]
        / grouped_close.shift(4)
        - 1.0
    )

    # Intrabar price range relative to the close.
    df["high_low_range"] = (
        df["high"] - df["low"]
    ) / df["close"]

    # Current-bar open-to-close return.
    df["close_open_return"] = (
        df["close"] / df["open"]
        - 1.0
    )

    # Percentage change in volume relative to
    # the immediately preceding 15-minute bar.
    previous_volume = grouped_volume.shift(1)

    df["volume_change"] = (
        df["volume"] / previous_volume
        - 1.0
    )

    return df