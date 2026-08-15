from datetime import datetime, time
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def to_ist(timestamp: datetime) -> datetime:
    """
    Convert a timezone-aware timestamp to Asia/Kolkata.

    Naive timestamps are assumed to already represent IST
    and are made timezone-aware.
    """

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=IST)

    return timestamp.astimezone(IST)


def is_normal_market_time(
    timestamp: datetime,
) -> bool:
    """
    Return True when timestamp falls inside the normal
    NSE/BSE continuous equity trading session.

    Weekends are excluded.
    """

    timestamp = to_ist(timestamp)

    if timestamp.weekday() >= 5:
        return False

    current_time = timestamp.time()

    return (
        MARKET_OPEN
        <= current_time
        <= MARKET_CLOSE
    )