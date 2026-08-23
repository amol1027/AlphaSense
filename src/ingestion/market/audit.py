from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from unittest import result

import pandas as pd

from src.features.market_time import (
    MARKET_OPEN,
    to_ist,
)
from src.features.trading_calendar import (
    TradingCalendar,
)


@dataclass
class MarketAuditResult:
    asset: str

    rows: int
    observed_sessions: int
    expected_sessions: int

    missing_sessions: list[date] = field(
        default_factory=list
    )
    unexpected_sessions: list[date] = field(
        default_factory=list
    )

    missing_bars: int = 0
    unexpected_bars: int = 0
    duplicate_timestamps: int = 0
    misaligned_timestamps: int = 0
    weekend_bars: int = 0
    holiday_bars: int = 0

    invalid_numeric_values: int = 0
    ohlc_violations: int = 0

    passed: bool = True

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _expected_session_timestamps(
    trading_date: date,
    session_windows: list[tuple[time, time]],
) -> set[pd.Timestamp]:
    """
    Return expected 15-minute candle timestamps for
    all trading windows on one session date.

    Timestamps are returned as UTC-aware pandas timestamps.
    """

    timestamps = set()

    for session_open, session_close in session_windows:
        current = datetime.combine(
            trading_date,
            session_open,
        )

        session_end = datetime.combine(
            trading_date,
            session_close,
        ) - timedelta(minutes=15)

        while current <= session_end:
            timestamps.add(
                pd.Timestamp(
                    to_ist(current)
                ).tz_convert("UTC")
            )

            current += timedelta(minutes=15)

    return timestamps

def audit_market_data(
    df: pd.DataFrame,
    asset: str,
    calendar: TradingCalendar,
) -> MarketAuditResult:
    """
    Audit one asset's 15-minute market dataset.

    This function does not modify the input data.
    """

    result = MarketAuditResult(
        asset=asset,
        rows=len(df),
        observed_sessions=0,
        expected_sessions=0,
    )

    if df.empty:
        result.passed = False
        return result

    timestamps = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    invalid_numeric = (
        df[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .isna()
        .any(axis=1)
    )

    result.invalid_numeric_values = int(
        invalid_numeric.sum()
    )

    result.duplicate_timestamps = int(
        timestamps.duplicated().sum()
    )

    valid_timestamps = timestamps.dropna()

    if valid_timestamps.empty:
        result.passed = False
        return result

    ist_timestamps = valid_timestamps.map(
        lambda value: value.tz_convert(
            "Asia/Kolkata"
        )
    )

    observed_dates = set(
        ist_timestamps.dt.date
    )

    observed_sessions = {
        trading_date
        for trading_date in observed_dates
        if calendar.is_trading_day(
            trading_date
        )
    }

    min_date = min(observed_dates)
    max_date = max(observed_dates)

    expected_dates = set()

    current_date = min_date

    while current_date <= max_date:
        if calendar.is_trading_day(
            current_date
        ):
            expected_dates.add(
                current_date
            )

        current_date += timedelta(days=1)

    result.observed_sessions = len(
        observed_sessions
    )

    result.expected_sessions = len(
        expected_dates
    )

    result.missing_sessions = sorted(
        expected_dates - observed_sessions
    )

    result.unexpected_sessions = sorted(
        observed_sessions - expected_dates
    )

    result.weekend_bars = int(sum(
        timestamp.weekday() >= 5
        and timestamp.date()
        not in getattr(
            calendar,
            "special_sessions",
            {},
        )
        for timestamp in ist_timestamps
    )
)

    result.holiday_bars = int(sum(
        not calendar.is_trading_day(
            timestamp.date()
        )
        for timestamp in ist_timestamps
    )
)

    expected_all = set()

    for trading_date in expected_dates:
        expected_all.update(
            _expected_session_timestamps(
                trading_date,
                calendar.session_windows(
                trading_date
            ),
        )
    )

    actual_all = set(
        valid_timestamps
    )

    result.missing_bars = len(
        expected_all - actual_all
    )

    result.unexpected_bars = len(
        actual_all - expected_all
    )

    misaligned = 0

    for timestamp in valid_timestamps:
        ist_timestamp = timestamp.tz_convert(
            "Asia/Kolkata"
        )

        if (
            ist_timestamp.minute % 15 != 0
            or ist_timestamp.second != 0
            or ist_timestamp.microsecond != 0
        ):
            misaligned += 1

    result.misaligned_timestamps = (
        misaligned
    )

    numeric = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    ohlc_invalid = (
        (numeric["open"] <= 0)
        | (numeric["high"] <= 0)
        | (numeric["low"] <= 0)
        | (numeric["close"] <= 0)
        | (numeric["volume"] < 0)
        | (numeric["high"] < numeric["low"])
        | (numeric["high"] < numeric["open"])
        | (numeric["high"] < numeric["close"])
        | (numeric["low"] > numeric["open"])
        | (numeric["low"] > numeric["close"])
    )

    result.ohlc_violations = int(
        ohlc_invalid.fillna(True).sum()
    )

    result.passed = all(
        [
            result.duplicate_timestamps == 0,
            result.invalid_numeric_values == 0,
            result.misaligned_timestamps == 0,
            result.weekend_bars == 0,
            result.holiday_bars == 0,
            result.missing_sessions == [],
            result.unexpected_sessions == [],
            result.missing_bars == 0,
            result.unexpected_bars == 0,
            result.ohlc_violations == 0,
        ]
    )

    return result