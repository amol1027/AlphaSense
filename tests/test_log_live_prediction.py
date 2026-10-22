import pandas as pd

from scripts.log_live_prediction import label_window_completes


def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts)


def test_morning_prediction_window_completes():
    # 10:00 IST -> forward bars through 11:00 IST (< 15:15 IST)
    assert label_window_completes(_utc("2026-10-21 04:30:00+00:00"))


def test_last_valid_slot_completes():
    # 14:15 IST -> forward bar opens land on 15:15 IST exactly
    assert label_window_completes(_utc("2026-10-21 08:45:00+00:00"))


def test_window_past_session_close_fails():
    # 14:30 IST -> needs the 15:30 IST bar, which does not exist
    assert not label_window_completes(_utc("2026-10-21 09:00:00+00:00"))


def test_naive_timestamp_treated_as_utc():
    assert label_window_completes(pd.Timestamp("2026-10-21 08:45:00"))
    assert not label_window_completes(pd.Timestamp("2026-10-21 09:00:00"))
