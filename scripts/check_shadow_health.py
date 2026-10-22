"""Daily self-check for the shadow run: token, cache freshness, log coverage.

Usage:
  python scripts/check_shadow_health.py

Scheduled shortly after the NSE close (e.g. 16:05 IST). Validates that:
  1. the Upstox token is still valid,
  2. the live cache was refreshed during the last trading session,
  3. the prediction log covered enough of that session's slots.

Any failure prints, emails (ALERT_EMAIL or SMTP_FROM recipient, optional), and
exits non-zero. Exit 0 never alerts.
"""

import json
import logging
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_upstox_token import token_status
from src.features.trading_calendar import NSEBSETradingCalendar
from src.features.trading_calendar import load_default_nse_bse_calendar
from src.notifications.smtp_email import SmtpEmailSender

IST = ZoneInfo("Asia/Kolkata")
LOG_PATH = PROJECT_ROOT / "data/interim/live_log.csv"
STATUS_PATH = PROJECT_ROOT / "data/interim/live_status.json"
ASSET_COUNT = 5
SLOT_MINUTES = 15
BLOCK_MINUTES = 75
MIN_COVERAGE = 0.5
COVERAGE_BUFFER = timedelta(minutes=30)
MAX_SESSION_LOOKBACK_DAYS = 15

logger = logging.getLogger("alphasense.health")


def expected_slot_times(session_open: time, session_close: time) -> list[time]:
    """Prediction slots t where bars at t+15..t+60 all exist before the close."""
    start = datetime.combine(date(2000, 1, 1), session_open)
    end_limit = datetime.combine(date(2000, 1, 1), session_close) - timedelta(
        minutes=BLOCK_MINUTES
    )
    slots = []
    current = start
    while current <= end_limit:
        slots.append(current.time())
        current += timedelta(minutes=SLOT_MINUTES)
    return slots


def last_trading_day(calendar: NSEBSETradingCalendar, today: date) -> date | None:
    for offset in range(MAX_SESSION_LOOKBACK_DAYS + 1):
        d = today - timedelta(days=offset)
        if calendar.is_trading_day(d):
            return d
    return None


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH)
    df["prediction_timestamp"] = pd.to_datetime(
        df["prediction_timestamp"], utc=True, format="mixed"
    )
    return df


def coverage_check(
    log: pd.DataFrame, calendar: NSEBSETradingCalendar, now: datetime
) -> tuple[bool | None, str]:
    """(None, msg) means deferred/not alertable; (bool, msg) is a verdict."""
    session_date = last_trading_day(calendar, now.date())
    if session_date is None:
        return (True, "no trading day found in lookback window")
    if session_date in calendar.special_sessions:
        return (None, f"session {session_date} is special; coverage skipped")
    session_open, session_close = calendar.session_windows(session_date)[0]
    close_ist = datetime.combine(session_date, session_close, IST)
    if now < close_ist + COVERAGE_BUFFER:
        return (None, f"session {session_date} still fresh; coverage deferred")
    if log.empty:
        rows = 0
    else:
        ist_dates = log["prediction_timestamp"].dt.tz_convert(IST).dt.date
        rows = int((ist_dates == session_date).sum())
    slots = expected_slot_times(session_open, session_close)
    expected = len(slots) * ASSET_COUNT
    ratio = rows / expected if expected else float("nan")
    ok = ratio >= MIN_COVERAGE
    return (
        ok,
        f"coverage {session_date}: {rows}/{expected} rows ({ratio:.0%}; slots "
        f"{session_open}-{session_close}; min target {MIN_COVERAGE:.0%})",
    )


def cache_freshness_check(
    status: dict | None, calendar: NSEBSETradingCalendar, now: datetime
) -> tuple[bool, str]:
    session_date = last_trading_day(calendar, now.date())
    if session_date is None:
        return (True, "no trading day found in lookback window")
    if not status or not status.get("last_refresh_at"):
        return (False, f"no refresh status recorded (last session {session_date})")
    refreshed = pd.Timestamp(status["last_refresh_at"]).tz_convert(IST)
    session_open = calendar.session_windows(session_date)[0][0]
    if refreshed.date() != session_date:
        return (
            False,
            f"last cache refresh {refreshed} is not from session {session_date}",
        )
    if refreshed.time() < session_open:
        return (False, f"cache refresh {refreshed} predates session open {session_open}")
    return (True, f"cache refreshed at {refreshed}")


def send_alert(subject: str, body: str) -> bool:
    recipient = os.getenv("ALERT_EMAIL", "") or os.getenv("SMTP_FROM", "")
    if not recipient:
        logger.info("No ALERT_EMAIL/SMTP_FROM configured; skipping email alert.")
        return False
    try:
        sender = SmtpEmailSender.from_env()
        sender.send(recipient, subject, body)
        logger.info("Alert email sent to %s.", recipient)
        return True
    except Exception:
        logger.exception("Alert email failed.")
        return False


def run_checks(
    calendar: NSEBSETradingCalendar,
    info: dict | None,
    token_ok: bool,
    token_message: str,
    log: pd.DataFrame,
    now: datetime,
) -> list[str]:
    failures = []
    checks = (
        ("TOKEN", token_ok, token_message),
        ("CACHE", *cache_freshness_check(info, calendar, now)),
        ("LOG", *coverage_check(log, calendar, now)),
    )
    for name, ok, message in checks:
        logger.info("%s: %s", name, message)
        if ok is False:
            failures.append(f"{name}: {message}")
    return failures


def load_status(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Could not parse live status.")
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    calendar = load_default_nse_bse_calendar()
    now = datetime.now(IST)
    token_ok, token_message = token_status()
    failures = run_checks(
        calendar,
        load_status(STATUS_PATH),
        token_ok,
        token_message,
        load_log(),
        now,
    )
    if failures:
        send_alert(
            "[AlphaSense] shadow health check FAILED",
            "Shadow health check failures:\n\n" + "\n".join(failures),
        )
        raise SystemExit(1)
    logger.info("Shadow health check passed.")


if __name__ == "__main__":
    main()
