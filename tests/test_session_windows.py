from datetime import datetime

from src.features.session_windows import (
    is_valid_prediction_timestamp,
    is_within_session,
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