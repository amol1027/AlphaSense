import json
from datetime import date
from pathlib import Path


class TradingCalendar:
    """
    Interface for exchange trading calendars.
    """

    def is_trading_day(
        self,
        trading_date: date,
    ) -> bool:
        raise NotImplementedError


class WeekdayTradingCalendar(
    TradingCalendar
):
    """
    Temporary calendar used for development
    and testing.

    Monday-Friday are treated as trading days.
    """

    def is_trading_day(
        self,
        trading_date: date,
    ) -> bool:
        return trading_date.weekday() < 5


class NSEBSETradingCalendar(
    TradingCalendar
):
    """
    Calendar for NSE/BSE normal equity trading.

    Holiday dates are supplied explicitly.
    """

    def __init__(
        self,
        holidays: set[date] | None = None,
    ):
        self.holidays = (
            holidays
            if holidays is not None
            else set()
        )

    def is_trading_day(
        self,
        trading_date: date,
    ) -> bool:

        if trading_date.weekday() >= 5:
            return False

        if trading_date in self.holidays:
            return False

        return True


def load_nse_bse_calendar(
    path: str | Path,
) -> NSEBSETradingCalendar:
    """
    Load an NSE/BSE normal-equity trading calendar
    from a JSON reference file.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    holidays = {
        date.fromisoformat(value)
        for value in data["holidays"]
    }

    return NSEBSETradingCalendar(
        holidays=holidays,
    )

DEFAULT_NSE_BSE_CALENDAR_PATH = (
    "data/reference/nse_bse_holidays_2026.json"
)


def load_default_nse_bse_calendar() -> NSEBSETradingCalendar:
    """
    Load the default NSE/BSE normal-equity
    trading calendar for the current project dataset.
    """

    return load_nse_bse_calendar(
        DEFAULT_NSE_BSE_CALENDAR_PATH
    )