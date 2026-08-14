import pandas as pd

from src.features.targets import add_next_hour_target


def test_next_hour_target():
    df = pd.DataFrame(
        {
            "asset": ["TCS", "TCS", "TCS"],
            "exchange": ["NSE", "NSE", "NSE"],
            "timestamp": pd.to_datetime(
                [
                    "2026-08-10 09:15:00",
                    "2026-08-10 10:15:00",
                    "2026-08-10 11:15:00",
                ]
            ),
            "close": [3505, 3520, 3530],
        }
    )

    result = add_next_hour_target(df)

    expected_return = (3530 - 3520) / 3520

    assert result.loc[1, "target_return"] == expected_return
    assert result.loc[1, "target_direction"] == 1
def test_missing_future_price_has_no_target():
    df = pd.DataFrame(
        {
            "asset": ["TCS", "TCS"],
            "exchange": ["NSE", "NSE"],
            "timestamp": pd.to_datetime(
                [
                    "2026-08-10 14:15:00",
                    "2026-08-10 15:15:00",
                ]
            ),
            "close": [3555, 3565],
        }
    )

    result = add_next_hour_target(df)

    assert pd.isna(result.loc[1, "target_return"])
    assert pd.isna(result.loc[1, "target_direction"])