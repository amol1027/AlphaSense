from datetime import datetime, timezone

from src.features.market_time import (
    is_normal_market_time,
    to_ist,
)


def test_naive_timestamp_is_assumed_ist():
    timestamp = datetime(
        2026,
        8,
        10,
        10,
        15,
    )

    result = to_ist(timestamp)

    assert result.tzinfo is not None
    assert result.hour == 10
    assert result.minute == 15


def test_utc_timestamp_is_converted_to_ist():
    timestamp = datetime(
        2026,
        8,
        10,
        4,
        45,
        tzinfo=timezone.utc,
    )

    result = to_ist(timestamp)

    assert result.hour == 10
    assert result.minute == 15


def test_market_open_is_valid():
    timestamp = datetime(
        2026,
        8,
        10,
        9,
        15,
    )

    assert is_normal_market_time(timestamp)


def test_market_close_is_valid():
    timestamp = datetime(
        2026,
        8,
        10,
        15,
        30,
    )

    assert is_normal_market_time(timestamp)


def test_before_market_open_is_invalid():
    timestamp = datetime(
        2026,
        8,
        10,
        9,
        14,
    )

    assert not is_normal_market_time(timestamp)


def test_after_market_close_is_invalid():
    timestamp = datetime(
        2026,
        8,
        10,
        15,
        31,
    )

    assert not is_normal_market_time(timestamp)


def test_weekend_is_invalid():
    timestamp = datetime(
        2026,
        8,
        15,
        10,
        15,
    )

    assert not is_normal_market_time(timestamp)