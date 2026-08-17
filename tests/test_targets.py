import pandas as pd

from src.features.targets import (
    add_next_hour_target,
)


def test_next_hour_target():
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
            "close": [
                3505,
                3510,
                3515,
                3518,
                3520,
            ],
        }
    )

    result = add_next_hour_target(df)

    expected_return = (
        (3520 - 3505) / 3505
    )

    assert (
        result.loc[0, "target_return"]
        == expected_return
    )

    assert (
        result.loc[0, "target_direction"]
        == 1
    )


def test_missing_future_price_has_no_target():
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
            "close": [
                3555,
                3560,
                3562,
                3565,
                3570,
            ],
        }
    )

    result = add_next_hour_target(df)

    assert pd.notna(
        result.loc[0, "target_return"]
    )

    assert pd.isna(
        result.loc[1, "target_return"]
    )

    assert pd.isna(
        result.loc[1, "target_direction"]
    )

def test_next_hour_target_with_15_minute_data():
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
            "close": [
                3505,
                3510,
                3515,
                3518,
                3520,
            ],
        }
    )

    result = add_next_hour_target(df)

    expected_return = (
        (3520 - 3505) / 3505
    )

    assert (
        result.loc[0, "target_return"]
        == expected_return
    )

    assert (
        result.loc[0, "target_direction"]
        == 1
    )

    assert pd.isna(
        result.loc[1, "target_return"]
    )