import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_market_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate canonical market OHLCV data.

    Returns a chronologically sorted copy.
    Raises ValueError when the data violates
    the canonical market-data contract.
    """

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required market columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            "Market data cannot be empty."
        )

    result = df[
        REQUIRED_COLUMNS
    ].copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="raise",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    if result["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate market timestamps found."
        )

    if result[
        numeric_columns
    ].isna().any().any():
        raise ValueError(
            "Market data contains missing numeric values."
        )

    if (
        result["open"] <= 0
    ).any() or (
        result["high"] <= 0
    ).any() or (
        result["low"] <= 0
    ).any() or (
        result["close"] <= 0
    ).any():
        raise ValueError(
            "OHLC prices must be positive."
        )

    if (
        result["volume"] < 0
    ).any():
        raise ValueError(
            "Volume cannot be negative."
        )

    if (
        result["high"] < result["low"]
    ).any():
        raise ValueError(
            "High price cannot be below low price."
        )

    if (
        result["high"] < result["open"]
    ).any() or (
        result["high"] < result["close"]
    ).any():
        raise ValueError(
            "High price must be >= open and close."
        )

    if (
        result["low"] > result["open"]
    ).any() or (
        result["low"] > result["close"]
    ).any():
        raise ValueError(
            "Low price must be <= open and close."
        )

    return result.sort_values(
        "timestamp"
    ).reset_index(drop=True)