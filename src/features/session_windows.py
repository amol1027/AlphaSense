from datetime import datetime, timedelta

from src.features.market_time import (
    MARKET_CLOSE,
    is_normal_market_time,
    to_ist,
)
from src.features.trading_calendar import (
    TradingCalendar,
    WeekdayTradingCalendar,
)


def is_within_session(
    timestamp: datetime,
    calendar: TradingCalendar | None = None,
) -> bool:
    """
    Return True when the timestamp falls inside the
    normal NSE/BSE continuous equity session and the
    date is a trading day.
    """

    calendar = (
        calendar
        if calendar is not None
        else WeekdayTradingCalendar()
    )

    timestamp = to_ist(timestamp)

    if not calendar.is_trading_day(
        timestamp.date()
    ):
        return False

    return is_normal_market_time(timestamp)


def is_valid_prediction_timestamp(
    prediction_timestamp: datetime,
    calendar: TradingCalendar | None = None,
) -> bool:
    """
    Return True when a complete one-hour prediction
    window fits inside the normal NSE/BSE session
    on a valid trading day.
    """

    calendar = (
        calendar
        if calendar is not None
        else WeekdayTradingCalendar()
    )

    prediction_timestamp = to_ist(
        prediction_timestamp
    )

    if not is_within_session(
        prediction_timestamp,
        calendar,
    ):
        return False

    window_end = (
        prediction_timestamp
        + timedelta(hours=1)
    )

    return (
        window_end.time()
        <= MARKET_CLOSE
    )