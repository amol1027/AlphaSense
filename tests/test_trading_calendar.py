from datetime import date, time
from src.features.trading_calendar import (
    load_nse_bse_calendar,)
from src.features.trading_calendar import (
    TradingCalendar,
    WeekdayTradingCalendar,)
from datetime import date

from src.features.trading_calendar import (
    NSEBSETradingCalendar,)


def test_load_nse_bse_calendar():
    calendar = load_nse_bse_calendar(
        "data/reference/nse_bse_holidays_2026.json"
    )

    assert not calendar.is_trading_day(
        date(2026, 1, 15)
    )

    assert not calendar.is_trading_day(
        date(2026, 1, 26)
    )

    assert calendar.is_trading_day(
        date(2026, 1, 16)
    )
def test_nse_bse_calendar_accepts_normal_weekday():
    calendar = NSEBSETradingCalendar()

    assert calendar.is_trading_day(
        date(2026, 8, 10)
    )


def test_nse_bse_calendar_rejects_weekend():
    calendar = NSEBSETradingCalendar()

    assert not calendar.is_trading_day(
        date(2026, 8, 15)
    )


def test_nse_bse_calendar_rejects_holiday():
    calendar = NSEBSETradingCalendar(
        holidays={
            date(2026, 8, 10),
        }
    )

    assert not calendar.is_trading_day(
        date(2026, 8, 10)
    )


def test_nse_bse_calendar_accepts_non_holiday_weekday():
    calendar = NSEBSETradingCalendar(
        holidays={
            date(2026, 8, 10),
        }
    )

    assert calendar.is_trading_day(
        date(2026, 8, 11)
    )


def test_weekday_is_trading_day():
    calendar = WeekdayTradingCalendar()

    monday = date(2026, 8, 10)

    assert calendar.is_trading_day(monday)


def test_saturday_is_not_trading_day():
    calendar = WeekdayTradingCalendar()

    saturday = date(2026, 8, 15)

    assert not calendar.is_trading_day(saturday)


def test_sunday_is_not_trading_day():
    calendar = WeekdayTradingCalendar()

    sunday = date(2026, 8, 16)

    assert not calendar.is_trading_day(sunday)


def test_base_calendar_requires_implementation():
    calendar = TradingCalendar()

    try:
        calendar.is_trading_day(
            date(2026, 8, 10)
        )
    except NotImplementedError:
        assert True
    else:
        assert False

def test_normal_session_windows():
    calendar = NSEBSETradingCalendar()

    assert calendar.session_windows(
        date(2026, 8, 10)
    ) == [
        (
            time(9, 15),
            time(15, 30),
        )
    ]


def test_special_weekend_session_is_trading_day():
    calendar = NSEBSETradingCalendar(
        special_sessions={
            date(2026, 2, 1): [
                (
                    time(9, 15),
                    time(15, 30),
                )
            ]
        }
    )

    assert calendar.is_trading_day(
        date(2026, 2, 1)
    )


def test_special_session_has_multiple_windows():
    calendar = NSEBSETradingCalendar(
        special_sessions={
            date(2024, 3, 2): [
                (
                    time(9, 15),
                    time(10, 0),
                ),
                (
                    time(11, 30),
                    time(12, 30),
                ),
            ]
        }
    )

    assert calendar.session_windows(
        date(2024, 3, 2)
    ) == [
        (
            time(9, 15),
            time(10, 0),
        ),
        (
            time(11, 30),
            time(12, 30),
        ),
    ]


def test_loaded_special_session_windows():
    calendar = load_nse_bse_calendar(
        "data/reference/nse_bse_holidays_2024.json"
    )

    assert calendar.session_windows(
        date(2024, 3, 2)
    ) == [
        (
            time(9, 15),
            time(10, 0),
        ),
        (
            time(11, 30),
            time(12, 30),
        ),
    ]