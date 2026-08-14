import pandas as pd

from src.features.build_features import build_hourly_features


def test_build_hourly_features():
    result = build_hourly_features(
        "data/raw/market_sample.csv",
        "data/raw/news_sample.csv",
    )

    tcs = result[
        (result["asset"] == "TCS")
        & (result["exchange"] == "NSE")
        & (
            result["prediction_timestamp"]
            == pd.Timestamp("2026-08-10 10:15:00")
        )
    ].iloc[0]

    assert tcs["close"] == 3520

    # Only 09:20 and 09:55 news are available
    # by the 10:15 prediction cutoff.
    assert tcs["news_count"] == 2

    # The next-hour target is based on
    # 10:15 close -> 11:15 close.
    expected_return = (3530 - 3520) / 3520

    assert tcs["target_return"] == expected_return
    assert tcs["target_direction"] == 1

def test_invalid_final_session_window_is_excluded():
    result = build_hourly_features(
        "data/raw/market_sample.csv",
        "data/raw/news_sample.csv",
    )

    invalid_rows = result[
        result["prediction_timestamp"]
        == pd.Timestamp("2026-08-10 15:15:00")
    ]

    assert len(invalid_rows) == 0