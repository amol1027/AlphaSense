"""Refresh 15-minute Upstox candles through the final NSE bar and log predictions."""

import logging
import time
from datetime import datetime, time as day_time
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
POLL_SECONDS = 15 * 60

logger = logging.getLogger("alphasense.shadow")


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
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
