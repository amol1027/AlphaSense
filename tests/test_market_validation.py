import pandas as pd
import pytest

from src.ingestion.market.validation import (
    validate_market_data,
)


def make_valid_data():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-08-10 10:15",
                    "2026-08-10 09:15",
                ]
            ),
            "open": [101, 100],
            "high": [103, 102],
            "low": [99, 98],
            "close": [102, 101],
            "volume": [2000, 1500],
        }
    )


def test_valid_market_data_is_sorted():
    result = validate_market_data(
        make_valid_data()
    )

    assert (
        result["timestamp"]
        .is_monotonic_increasing
    )


def test_missing_columns_are_rejected():
    df = make_valid_data().drop(
        columns=["volume"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required market columns",
    ):
        validate_market_data(df)


def test_empty_data_is_rejected():
    df = pd.DataFrame(
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        validate_market_data(df)


def test_duplicate_timestamps_are_rejected():
    df = make_valid_data()

    df.loc[1, "timestamp"] = df.loc[
        0, "timestamp"
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate market timestamps",
    ):
        validate_market_data(df)


def test_negative_volume_is_rejected():
    df = make_valid_data()

    df.loc[0, "volume"] = -1

    with pytest.raises(
        ValueError,
        match="Volume cannot be negative",
    ):
        validate_market_data(df)


def test_non_positive_prices_are_rejected():
    df = make_valid_data()

    df.loc[0, "close"] = 0

    with pytest.raises(
        ValueError,
        match="OHLC prices must be positive",
    ):
        validate_market_data(df)


def test_high_below_low_is_rejected():
    df = make_valid_data()

    df.loc[0, "high"] = 90

    with pytest.raises(
        ValueError,
        match="High price cannot be below low",
    ):
        validate_market_data(df)


def test_high_below_close_is_rejected():
    df = make_valid_data()

    df.loc[0, "high"] = 100

    with pytest.raises(
        ValueError,
        match="High price must be >= open and close",
    ):
        validate_market_data(df)


def test_low_above_close_is_rejected():
    df = make_valid_data()

    df.loc[0, "low"] = 103

    with pytest.raises(
        ValueError,
        match="Low price must be <= open and close",
    ):
        validate_market_data(df)