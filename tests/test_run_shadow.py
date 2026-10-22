"""Scheduling tests for the boundary-aligned shadow runner."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scripts.run_shadow import seconds_until_next_cycle

IST = ZoneInfo("Asia/Kolkata")


def _at(day, month, year, hour, minute, second=0):
    return datetime(year, month, day, hour, minute, second, tzinfo=IST)


def test_mid_session_wakes_after_next_boundary():
    # Monday 10:07 -> 10:15 boundary + 120s = 10:17
    assert seconds_until_next_cycle(_at(5, 10, 2026, 10, 7)) == pytest.approx(600.0)


def test_inside_completion_delay_wakes_at_delay_end():
    # Monday 10:15:30 -> 10:17:00
    assert seconds_until_next_cycle(_at(5, 10, 2026, 10, 15, 30)) == pytest.approx(90.0)


def test_after_delay_waits_for_next_boundary():
    # Monday 10:18 -> 10:30 boundary + 120s = 10:32
    assert seconds_until_next_cycle(_at(5, 10, 2026, 10, 18)) == pytest.approx(840.0)


def test_late_session_preserves_final_close_cycle():
    # Monday 15:44 -> final wake at 15:45 close
    assert seconds_until_next_cycle(_at(5, 10, 2026, 15, 44)) == pytest.approx(60.0)


def test_after_close_sleeps_until_next_open():
    # Monday 16:00 -> Tuesday 09:15 (17h15m)
    assert seconds_until_next_cycle(_at(5, 10, 2026, 16, 0)) == pytest.approx(
        17 * 3600 + 15 * 60
    )


def test_pre_open_wakes_at_today_open():
    # Monday 08:00 -> 09:15 same day
    assert seconds_until_next_cycle(_at(5, 10, 2026, 8, 0)) == pytest.approx(4500.0)


def test_friday_evening_skips_weekend():
    # Friday 16:00 -> Monday 09:15
    assert seconds_until_next_cycle(_at(2, 10, 2026, 16, 0)) == pytest.approx(
        65 * 3600 + 15 * 60
    )


def test_saturday_sleeps_until_monday_open():
    # Saturday 10:00 -> Monday 09:15
    assert seconds_until_next_cycle(_at(3, 10, 2026, 10, 0)) == pytest.approx(
        47 * 3600 + 15 * 60
    )
