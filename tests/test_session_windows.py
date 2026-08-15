from datetime import datetime
from datetime import date
from src.features.trading_calendar import (
    WeekdayTradingCalendar,)
from src.features.session_windows import (
    is_valid_prediction_timestamp,
    is_within_session,)

class HolidayCalendar(WeekdayTradingCalendar):
    def __init__(self, holidays):
        self.holidays = set(holidays)

    def is_trading_day(self, trading_date):
        if trading_date in self.holidays:
            return False

        return super().is_trading_day(
            trading_date
        )
def test_holiday_is_not_valid_prediction_day():
    calendar = HolidayCalendar(
        {
            date(2026, 8, 10),
        }
    )

    timestamp = datetime(
        2026,
        8,
        10,
        10,
        15,
    )

    assert not is_valid_prediction_timestamp(
        timestamp,
        calendar,
    )

def test_non_holiday_weekday_remains_valid():
    calendar = HolidayCalendar(
        {
            date(2026, 8, 10),
        }
    )

    timestamp = datetime(
        2026,
        8,
        11,
        10,
        15,
    )

    assert is_valid_prediction_timestamp(
        timestamp,
        calendar,
    )

def test_timestamp_inside_session():
    timestamp = datetime(
        2026,
        8,
        10,
        10,
        15,
    )

    assert is_within_session(timestamp)


def test_timestamp_before_session():
    timestamp = datetime(
        2026,
        8,
        10,
        9,
        0,
    )

    assert not is_within_session(timestamp)


def test_timestamp_after_session():
    timestamp = datetime(
        2026,
        8,
        10,
        16,
        0,
    )

    assert not is_within_session(timestamp)


def test_weekend_is_not_session():
    timestamp = datetime(
        2026,
        8,
        8,
        10,
        15,
    )

    assert not is_within_session(timestamp)


def test_complete_hour_is_valid():
    timestamp = datetime(
        2026,
        8,
        10,
        14,
        15,
    )

    assert is_valid_prediction_timestamp(
        timestamp
    )


def test_last_partial_hour_is_invalid():
    timestamp = datetime(
        2026,
        8,
        10,
        15,
        15,
    )

    assert not is_valid_prediction_timestamp(
        timestamp
    )