from datetime import date

import pandas as pd

from src.features.trading_calendar import (
    NSEBSETradingCalendar,
)
from src.ingestion.market.audit import (
    audit_market_data,
)


def make_valid_session():
    timestamps = pd.date_range(
        "2026-08-10 03:45:00+00:00",
        periods=25,
        freq="15min",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * 25,
            "high": [101.0] * 25,
            "low": [99.0] * 25,
            "close": [100.5] * 25,
            "volume": [1000] * 25,
        }
    )


def test_valid_session_passes():
    df = make_valid_session()

    calendar = NSEBSETradingCalendar()

    result = audit_market_data(
        df,
        asset="TCS",
        calendar=calendar,
    )

    assert result.rows == 25
    assert result.observed_sessions == 1
    assert result.expected_sessions == 1
    assert result.missing_bars == 0
    assert result.unexpected_bars == 0
    assert result.passed is True


def test_missing_bar_is_detected():
    df = make_valid_session().drop(
        index=10
    ).reset_index(drop=True)

    calendar = NSEBSETradingCalendar()

    result = audit_market_data(
        df,
        asset="TCS",
        calendar=calendar,
    )

    assert result.missing_bars == 1
    assert result.passed is False


def test_duplicate_timestamp_is_detected():
    df = make_valid_session()

    df = pd.concat(
        [
            df,
            df.iloc[[0]],
        ],
        ignore_index=True,
    )

    calendar = NSEBSETradingCalendar()

    result = audit_market_data(
        df,
        asset="TCS",
        calendar=calendar,
    )

    assert result.duplicate_timestamps == 1
    assert result.passed is False


def test_holiday_bar_is_detected():
    df = make_valid_session()

    calendar = NSEBSETradingCalendar(
        holidays={
            date(2026, 8, 10)
        }
    )

    result = audit_market_data(
        df,
        asset="TCS",
        calendar=calendar,
    )

    assert result.holiday_bars == 25
    assert result.passed is False