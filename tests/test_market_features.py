import pandas as pd
import pytest

from src.features.market_features import (
    add_normalized_market_features,
)


def test_normalized_market_features():
    df = pd.DataFrame(
        {
            "asset": ["TCS"] * 5,
            "exchange": ["NSE"] * 5,
            "timestamp": pd.to_datetime(
                [
                    "2026-08-10 09:15:00",
                    "2026-08-10 09:30:00",
                    "2026-08-10 09:45:00",
                    "2026-08-10 10:00:00",
                    "2026-08-10 10:15:00",
                ]
            ),
            "open": [
                100,
                101,
                102,
                103,
                104,
            ],
            "high": [
                102,
                103,
                104,
                105,
                106,
            ],
            "low": [
                99,
                100,
                101,
                102,
                103,
            ],
            "close": [
                101,
                102,
                103,
                104,
                105,
            ],
            "volume": [
                1000,
                1100,
                1200,
                1300,
                1400,
            ],
        }
    )

    result = add_normalized_market_features(
        df
    )

    # First row has no previous observation.
    assert pd.isna(
        result.loc[0, "return_15m"]
    )

    # At 10:15, the 15-minute return is:
    # (105 / 104) - 1.
    assert result.loc[
        4, "return_15m"
    ] == pytest.approx(
        (105 / 104) - 1
    )

    # At 10:15, the 30-minute return is:
    # (105 / 103) - 1.
    assert result.loc[
        4, "return_30m"
    ] == pytest.approx(
        (105 / 103) - 1
    )

    # At 10:15, the 1-hour return is:
    # (105 / 101) - 1.
    assert result.loc[
        4, "return_1h"
    ] == pytest.approx(
        (105 / 101) - 1
    )

    assert result.loc[
        4, "high_low_range"
    ] == pytest.approx(
        (106 - 103) / 105
    )

    assert result.loc[
        4, "close_open_return"
    ] == pytest.approx(
        (105 / 104) - 1
    )

    assert result.loc[
        4, "volume_change"
    ] == pytest.approx(
        (1400 / 1300) - 1
    )


def test_normalized_features_are_grouped_by_asset():
    df = pd.DataFrame(
        {
            "asset": [
                "TCS",
                "TCS",
                "RELIANCE",
                "RELIANCE",
            ],
            "exchange": [
                "NSE",
                "NSE",
                "NSE",
                "NSE",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-08-10 09:15:00",
                    "2026-08-10 09:30:00",
                    "2026-08-10 09:15:00",
                    "2026-08-10 09:30:00",
                ]
            ),
            "open": [
                100,
                101,
                200,
                202,
            ],
            "high": [
                102,
                103,
                203,
                205,
            ],
            "low": [
                99,
                100,
                199,
                201,
            ],
            "close": [
                101,
                102,
                202,
                204,
            ],
            "volume": [
                1000,
                1100,
                2000,
                2200,
            ],
        }
    )

    result = add_normalized_market_features(
        df
    )

    # The first observation of each asset must
    # have no lagged return.
    assert pd.isna(
        result.loc[
            result["asset"] == "TCS",
            "return_15m",
        ].iloc[0]
    )

    assert pd.isna(
        result.loc[
            result["asset"] == "RELIANCE",
            "return_15m",
        ].iloc[0]
    )