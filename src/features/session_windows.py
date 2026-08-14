from datetime import datetime, time, timedelta


SESSION_START = time(9, 15)
SESSION_END = time(15, 30)


def is_within_session(timestamp: datetime) -> bool:
    """
    Return True when timestamp falls inside the regular
    NSE/BSE equity trading session.

    This initial Phase 0 implementation uses a simplified
    weekday session model. Exchange holidays will be handled
    by a proper market calendar later.
    """

    if timestamp.weekday() >= 5:
        return False

    current_time = timestamp.time()

    return SESSION_START <= current_time <= SESSION_END


def is_valid_prediction_timestamp(
    prediction_timestamp: datetime,
) -> bool:
    """
    Return True when a full one-hour prediction window
    fits inside the regular trading session.
    """

    if not is_within_session(prediction_timestamp):
        return False

    window_end = (
        prediction_timestamp
        + timedelta(hours=1)
    )

    return window_end.time() <= SESSION_END