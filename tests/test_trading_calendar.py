from datetime import date

from src.features.trading_calendar import (
    TradingCalendar,
    WeekdayTradingCalendar,
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