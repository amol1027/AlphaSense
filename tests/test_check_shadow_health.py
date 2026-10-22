import pandas as pd

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from scripts.check_shadow_health import (
    MIN_COVERAGE,
    coverage_check,
    cache_freshness_check,
    expected_slot_times,
)
from src.features.trading_calendar import NSEBSETradingCalendar

IST = ZoneInfo("Asia/Kolkata")

# 2026-10-22 (Thu) trading; Friday holiday; Monday trading; no weekend sessions.
CALENDAR = NSEBSETradingCalendar(holidays={date(2026, 10, 23), date(2026, 10, 26)})


def _now(ts: str) -> datetime:
    return pd.Timestamp(ts, tz="Asia/Kolkata").to_pydatetime()


def _log(timestamps) -> pd.DataFrame:
    return pd.DataFrame(
        {"prediction_timestamp": pd.to_datetime(timestamps, utc=True, format="mixed")}
    )


def test_expected_slots_normal_session():
    slots = expected_slot_times(time(9, 15), time(15, 30))
    assert slots[0] == time(9, 15)
    assert slots[-1] == time(14, 15)
    assert len(slots) == 21


def test_expected_slots_short_session():
    # Diwali-style shortened session: 09:15 -> 13:45 => last slot 12:30
    slots = expected_slot_times(time(9, 15), time(13, 45))
    assert slots[-1] == time(12, 30)
    assert len(slots) == 14


def test_expected_slots_are_regular_15m_steps():
    slots = expected_slot_times(time(9, 15), time(15, 30))
    assert [s.minute for s in slots[:2]] == [15, 30]


def test_cache_fresh_missing_status_fails():
    ok, message = cache_freshness_check(None, CALENDAR, _now("2026-10-22 16:05"))
    assert not ok


def test_cache_fresh_today_session_passes():
    status = {"last_refresh_at": "2026-10-22 07:30:00+00:00"}
    ok, message = cache_freshness_check(status, CALENDAR, _now("2026-10-22 16:05"))
    assert ok


def test_cache_fresh_stale_from_previous_session_fails():
    # Friday holiday -> last session is Thursday 2026-10-22; a refresh from
    # Monday 2026-10-19 is stale.
    status = {"last_refresh_at": "2026-10-19 07:00:00+00:00"}
    ok, message = cache_freshness_check(status, CALENDAR, _now("2026-10-23 16:05"))
    assert not ok


def test_cache_before_session_open_fails():
    status = {"last_refresh_at": "2026-10-22 03:00:00+00:00"}  # 08:30 IST
    ok, message = cache_freshness_check(status, CALENDAR, _now("2026-10-22 16:05"))
    assert not ok


def test_coverage_full_session_passes():
    # Thursday session 09:15-15:30; 21 slots * 5 assets after close+buffer.
    stamps = [
        pd.Timestamp("2026-10-22 04:00:00+00:00") + pd.Timedelta(minutes=15 * i)
        for i in range(21)
        for _ in range(5)
    ]
    ok, _ = coverage_check(
        _log(stamps),
        CALENDAR,
        _now("2026-10-22 16:05"),
    )
    assert ok is True


def test_coverage_near_empty_session_fails():
    stamps = [pd.Timestamp("2026-10-22 04:00:00+00:00")] * 5
    ok, message = coverage_check(_log(stamps), CALENDAR, _now("2026-10-22 16:05"))
    assert ok is False


def test_coverage_below_half_session_covers_alert_boundary():
    stamps = [
        pd.Timestamp("2026-10-22 04:00:00+00:00") + pd.Timedelta(minutes=15 * i)
        for i in range(10)
        for _ in range(5)
    ]
    ok, _ = coverage_check(_log(stamps), CALENDAR, _now("2026-10-22 16:05"))
    assert ok == (10 / 21 >= MIN_COVERAGE)


def test_coverage_deferred_before_buffer():
    ok, message = coverage_check(pd.DataFrame(), CALENDAR, _now("2026-10-22 14:00"))
    assert ok is None


def test_coverage_skipped_on_special_session():
    calendar = NSEBSETradingCalendar(
        holidays={date(2026, 10, 23), date(2026, 10, 26)},
        special_sessions={date(2026, 10, 22): [(time(9, 15), time(13, 45))]},
    )
    ok, message = coverage_check(
        pd.DataFrame(), calendar, _now("2026-10-22 16:05"),
    )
    assert ok is None


def test_coverage_monday_after_weekend_and_friday_holiday():
    # Sunday 2026-10-25: Friday holiday; last session Thursday 2026-10-22.
    stamps = [
        pd.Timestamp("2026-10-22 04:00:00+00:00") + pd.Timedelta(minutes=15 * i)
        for i in range(21)
        for _ in range(5)
    ]
    ok, _ = coverage_check(_log(stamps), CALENDAR, _now("2026-10-25 16:05"))
    assert ok is True
