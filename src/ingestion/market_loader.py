import pandas as pd


REQUIRED_COLUMNS = {
    "asset",
    "exchange",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def load_market_data(path: str) -> pd.DataFrame:
    """Load and validate market OHLCV data."""

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    df = df.sort_values(
        ["asset", "exchange", "timestamp"]
    ).reset_index(drop=True)

    return df