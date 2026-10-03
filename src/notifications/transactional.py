"""Transactional account emails: welcome + sign-in alerts.

These are security emails, not regime reminders: they are sent whenever
SMTP is configured, regardless of the user's reminder preferences, and
are always best-effort (a provider failure must never break register or
login — callers must swallow all exceptions).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger("alphasense.notifications")

IST = ZoneInfo("Asia/Kolkata")


def _ist_short(moment: datetime | None = None) -> str:
    moment = moment or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(IST).strftime("%d %b %Y, %H:%M IST")


def format_welcome_email(display_name: str) -> tuple[str, str]:
    first = (display_name or "").split(" ")[0] or display_name
    subject = "Welcome to AlphaSense research"
    body = (
        f"Hi {first},\n\n"
        "Your AlphaSense account is ready. This site publishes NSE/BSE "
        "volatility-regime research; your account holds only your name, "
        "email, and reminder preferences.\n\n"
        "For your security, we email you on every sign-in. If you ever get "
        "a sign-in alert you do not recognise, change your password and "
        "sign out other sessions from your profile.\n\n"
        "Research prototype, not trading advice."
    )
    return subject, body


def format_login_alert_email(display_name: str, ip: str | None, moment: datetime | None = None) -> tuple[str, str]:
    first = (display_name or "").split(" ")[0] or display_name
    subject = "AlphaSense sign-in alert"
    body = (
        f"Hi {first},\n\n"
        f"Your AlphaSense account just signed in at {_ist_short(moment)}"
        f"{f' from IP {ip}' if ip and ip != 'unknown' else ''}.\n\n"
        "If this was you, no action is needed. If you do not recognise this "
        "sign-in, change your password immediately and sign out other "
        "sessions from your profile."
    )
    return subject, body


def send_security_email(to: str, subject: str, body: str) -> bool:
    """Send one transactional email. Returns True on success.

    Returns False (never raises) when SMTP is unconfigured or sending
    fails, so auth request paths stay unaffected.
    """
    if not to or "@" not in to:
        return False
    if not os.getenv("SMTP_HOST"):
        return False
    try:
        from src.notifications.smtp_email import SmtpEmailSender

        sender = SmtpEmailSender.from_env()
        sender.send(to, subject, body)
    except Exception:
        logger.warning("security email send failed", exc_info=True)
        return False
    return True
