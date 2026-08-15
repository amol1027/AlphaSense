import pandas as pd

from src.features.build_features import build_hourly_features
from datetime import date

from src.features.trading_calendar import (
    WeekdayTradingCalendar,
)


class HolidayCalendar(WeekdayTradingCalendar):
    def __init__(self, holidays):
        self.holidays = set(holidays)

    def is_trading_day(self, trading_date):
        if trading_date in self.holidays:
            return False

        return super().is_trading_day(
            trading_date
        )


def test_build_features_respects_holiday_calendar():
    calendar = HolidayCalendar(
        {
            date(2026, 8, 10),
        }
    )

    result = build_hourly_features(
        "data/raw/market_sample.csv",
        "data/raw/news_sample.csv",
        "data/raw/reddit_sample.csv",
        calendar=calendar,
    )

    holiday_rows = result[
        result["prediction_timestamp"].dt.date
        == date(2026, 8, 10)
    ]

    assert len(holiday_rows) == 0

def test_build_hourly_features():
    result = build_hourly_features(
    "data/raw/market_sample.csv",
    "data/raw/news_sample.csv",
    "data/raw/reddit_sample.csv",
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
        "data/raw/reddit_sample.csv",
    )

    invalid_rows = result[
        result["prediction_timestamp"]
        == pd.Timestamp("2026-08-10 15:15:00")
    ]

    assert len(invalid_rows) == 0

def test_reddit_features_are_integrated():

    result = build_hourly_features(
        "data/raw/market_sample.csv",
        "data/raw/news_sample.csv",
        "data/raw/reddit_sample.csv",
    )

    tcs = result[
        (result["asset"] == "TCS")
        & (result["exchange"] == "NSE")
        & (
            result["prediction_timestamp"]
            == pd.Timestamp(
                "2026-08-10 10:15:00"
            )
        )
    ].iloc[0]

    # Only Reddit posts published by 10:15
    # should be included.
    assert tcs["reddit_count"] == 2

    # The 10:20 Reddit post must NOT appear.
    assert tcs["reddit_count"] != 3