"""Refresh 15-minute Upstox candles through the final NSE bar and log predictions."""

import logging
import time
from datetime import datetime, time as day_time, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_upstox_token import token_status
from scripts.log_live_prediction import cmd_label, cmd_log
from scripts.refresh_live_market import refresh

IST = ZoneInfo("Asia/Kolkata")
OPEN = day_time(9, 15)
# Allow a final cycle after the 15:30 close so the 15:15–15:30 candle is complete.
CLOSE = day_time(15, 45)
CYCLE_MINUTES = 15
# Wakes land just after each 15m bar boundary so the freshly completed candle
# is already available instead of polling on a fixed sleep from boot time.
COMPLETION_DELAY_SECONDS = 120
# Brief pause before recomputing the schedule after a failed cycle.
RETRY_SECONDS = 60

logger = logging.getLogger("alphasense.shadow")


def _next_weekday_open(now: datetime) -> datetime:
    """Next 09:15 IST session start (today if still ahead, else next weekday)."""
    opening = now.replace(hour=OPEN.hour, minute=OPEN.minute, second=0, microsecond=0)
    if now.weekday() < 5 and now < opening:
        return opening
    nxt = now + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.replace(hour=OPEN.hour, minute=OPEN.minute, second=0, microsecond=0)


def seconds_until_next_cycle(now: datetime) -> float:
    """Seconds from `now` (IST-aware) until the next scheduled wake.

    In-session wakes land on the next 15m bar boundary plus the completion
    delay; a final wake at CLOSE preserves the end-of-session cycle;
    otherwise sleeps until the next weekday session open.
    """
    if now.weekday() < 5:
        day_open = now.replace(hour=OPEN.hour, minute=OPEN.minute, second=0, microsecond=0)
        day_close = now.replace(hour=CLOSE.hour, minute=CLOSE.minute, second=0, microsecond=0)
        if day_open <= now <= day_close:
            floored = now.replace(second=0, microsecond=0)
            last_boundary = floored.replace(
                minute=(floored.minute // CYCLE_MINUTES) * CYCLE_MINUTES
            )
            candidate = last_boundary + timedelta(seconds=COMPLETION_DELAY_SECONDS)
            if candidate <= now + timedelta(seconds=5):
                candidate += timedelta(minutes=CYCLE_MINUTES)
            if candidate <= day_close:
                return max((candidate - now).total_seconds(), 1.0)
            if now < day_close:
                return max((day_close - now).total_seconds(), 1.0)
    target = _next_weekday_open(now)
    return max((target - now).total_seconds(), 1.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ok, message = token_status()
    if not ok:
        # Fail loud at boot: without a valid token every cycle would burn
        # 15 minutes serving stale bars. Renew via manual OAuth and restart.
        logger.error("Refusing to start: %s", message)
        raise SystemExit(1)
    logger.info("Shadow runner started (%s); Ctrl+C to stop.", message)
    failures = 0
    while True:
        now = datetime.now(IST)
        if now.weekday() < 5 and OPEN <= now.time() <= CLOSE:
            try:
                refresh()
                cmd_label()
                cmd_log()
                failures = 0
            except Exception:
                failures += 1
                logger.exception("Shadow cycle failed (%d consecutive)", failures)
                time.sleep(RETRY_SECONDS)
                continue
        time.sleep(seconds_until_next_cycle(datetime.now(IST)))


if __name__ == "__main__":
    main()
