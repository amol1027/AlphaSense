from datetime import date


class TradingCalendar:
    """
    Abstract trading-calendar interface.

    The initial implementation is intentionally simple.
    A production NSE/BSE holiday calendar can be plugged in later.
    """

    def is_trading_day(self, trading_date: date) -> bool:
        """
        Return True when the exchange is open for normal trading
        on the given date.
        """

        raise NotImplementedError


class WeekdayTradingCalendar(TradingCalendar):
    """
    Temporary Phase 0 implementation.

    Monday-Friday are considered trading days.
    Exchange holidays will be supplied by a real calendar later.
    """

    def is_trading_day(self, trading_date: date) -> bool:
        return trading_date.weekday() < 5