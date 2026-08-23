import json
from datetime import date, time
from pathlib import Path


SessionWindow = tuple[time, time]


class TradingCalendar:
    """
    Interface for exchange trading calendars.
    """

    def is_trading_day(
        self,
        trading_date: date,
    ) -> bool:
        raise NotImplementedError

    def session_windows(
        self,
        trading_date: date,
    ) -> list[SessionWindow]:
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

    def session_windows(
        self,
        trading_date: date,
    ) -> list[SessionWindow]:
        if not self.is_trading_day(
            trading_date
        ):
            return []

        return [
            (
                time(9, 15),
                time(15, 30),
            )
        ]


class NSEBSETradingCalendar(
    TradingCalendar
):
    """
    Calendar for NSE/BSE normal equity trading.

    Holiday dates and special session windows
    are supplied explicitly.
    """

    def __init__(
        self,
        holidays: set[date] | None = None,
        special_sessions: (
            dict[date, list[SessionWindow]]
            | None
        ) = None,
    ):
        self.holidays = (
            holidays
            if holidays is not None
            else set()
        )

        self.special_sessions = (
            special_sessions
            if special_sessions is not None
            else {}
        )

    def is_trading_day(
        self,
        trading_date: date,
    ) -> bool:

        if trading_date in self.special_sessions:
            return True

        if trading_date.weekday() >= 5:
            return False

        if trading_date in self.holidays:
            return False

        return True

    def session_windows(
        self,
        trading_date: date,
    ) -> list[SessionWindow]:

        if trading_date in self.special_sessions:
            return self.special_sessions[
                trading_date
            ]

        if not self.is_trading_day(
            trading_date
        ):
            return []

        return [
            (
                time(9, 15),
                time(15, 30),
            )
        ]


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

    special_sessions = {}

    for date_value, session_data in (
        data.get(
            "special_sessions",
            {},
        ).items()
    ):
        windows = []

        for window in session_data["sessions"]:
            windows.append(
                (
                    time.fromisoformat(
                        window["open"]
                    ),
                    time.fromisoformat(
                        window["close"]
                    ),
                )
            )

        special_sessions[
            date.fromisoformat(date_value)
        ] = windows

    return NSEBSETradingCalendar(
        holidays=holidays,
        special_sessions=special_sessions,
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