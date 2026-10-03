"""Best-effort fan-out: predictions -> opted-in users -> email/WhatsApp.

Never raises for provider failures; each failure is logged and recorded
as status='failed' in notification_log so a later run can retry via a new
dedupe key (prediction_timestamp moves on). Dedupe is enforced by the
UNIQUE (dedupe_key, channel) key: re-runs never double-send.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.notifications.rules import (
    build_dedupe_key,
    format_email,
    format_whatsapp_params,
    normalize_assets,
    should_notify,
)

logger = logging.getLogger("alphasense.notifications")

WHATSAPP_TEMPLATE_ENV = "WHATSAPP_TEMPLATE"


def _parse_assets(raw: Any) -> list[str]:
    if not raw:
        from src.notifications.rules import KNOWN_ASSETS

        return list(KNOWN_ASSETS)
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (ValueError, TypeError):
        from src.notifications.rules import KNOWN_ASSETS

        return list(KNOWN_ASSETS)
    return normalize_assets(parsed)


def get_subscribers(db) -> list[dict]:
    """Load enabled subscribers with normalized prefs. DB-missing -> []."""
    try:
        rows = db.fetch_all(
            "SELECT u.id, u.email, u.whatsapp_e164, p.email_enabled, "
            "p.whatsapp_enabled, p.assets, p.min_probability, p.daily_summary "
            "FROM users u LEFT JOIN notification_prefs p ON p.user_id = u.id "
            "WHERE u.disabled_at IS NULL",
            (),
        )
    except Exception:
        logger.warning("subscriber load failed (is 002_notifications.sql applied?)", exc_info=True)
        return []
    subscribers = []
    for row in rows or []:
        email_enabled = bool(row.get("email_enabled", 1))
        whatsapp_enabled = bool(row.get("whatsapp_enabled", 0))
        if not email_enabled and not whatsapp_enabled:
            continue
        subscribers.append(
            {
                "user_id": int(row["id"]),
                "email": row.get("email"),
                "whatsapp_e164": row.get("whatsapp_e164"),
                "prefs": {
                    "email_enabled": email_enabled,
                    "whatsapp_enabled": whatsapp_enabled,
                    "whatsapp_e164": row.get("whatsapp_e164"),
                    "assets": _parse_assets(row.get("assets")),
                    "min_probability": row.get("min_probability", 0.70) or 0.70,
                    "daily_summary": bool(row.get("daily_summary", 0)),
                },
            }
        )
    return subscribers


def _already_sent(db, dedupe_key: str, channel: str) -> bool:
    try:
        row = db.fetch_one(
            "SELECT id FROM notification_log WHERE dedupe_key = %s AND channel = %s",
            (dedupe_key, channel),
        )
    except Exception:
        return False
    return row is not None


def _record(db, user_id: int, channel: str, asset: str, dedupe_key: str, status: str, provider_id: str | None) -> None:
    try:
        db.execute(
            "INSERT INTO notification_log (user_id, channel, type, asset, dedupe_key, status, provider_msg_id) "
            "VALUES (%s, %s, 'high_range', %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE status = VALUES(status)",
            (user_id, channel, asset, dedupe_key, status, provider_id),
        )
    except Exception:
        logger.warning("notification_log write failed", exc_info=True)


def _probability_of(prediction: dict) -> float:
    try:
        return float(prediction.get("probability_high_range", prediction.get("probability", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def send_for_snapshot(
    predictions: dict,
    *,
    db,
    email_sender=None,
    whatsapp_sender=None,
    whatsapp_template: str = "volatility_alert",
    moment: datetime | None = None,
    dry_run: bool = False,
) -> dict:
    """Fan out one dashboard_snapshot()['predictions'] dict. Returns counts."""
    now = moment or datetime.now(timezone.utc)
    summary = {"subscribers": 0, "considered": 0, "sent_email": 0, "sent_whatsapp": 0, "skipped": 0, "failed": 0}
    subscribers = get_subscribers(db)
    summary["subscribers"] = len(subscribers)
    if not subscribers:
        return summary
    for subscriber in subscribers:
        for asset, prediction in (predictions or {}).items():
            if not isinstance(prediction, dict) or prediction.get("error"):
                continue
            summary["considered"] += 1
            ok, reason = should_notify(prediction, subscriber["prefs"], now)
            if not ok:
                summary["skipped"] += 1
                continue
            proba = _probability_of(prediction)
            ts = str(prediction.get("prediction_timestamp", now.isoformat()))
            # Email
            if subscriber["prefs"]["email_enabled"] and email_sender is not None and subscriber.get("email"):
                key = build_dedupe_key(subscriber["user_id"], asset, ts)
                if _already_sent(db, key, "email"):
                    summary["skipped"] += 1
                elif dry_run:
                    summary["sent_email"] += 1
                else:
                    try:
                        subject, body = format_email(asset, proba, ts)
                        msg_id = email_sender.send(subscriber["email"], subject, body)
                        _record(db, subscriber["user_id"], "email", asset, key, "sent", str(msg_id))
                        summary["sent_email"] += 1
                    except Exception:
                        logger.warning("email send failed for user %s", subscriber["user_id"], exc_info=True)
                        _record(db, subscriber["user_id"], "email", asset, key, "failed", None)
                        summary["failed"] += 1
            # WhatsApp
            if (
                subscriber["prefs"]["whatsapp_enabled"]
                and whatsapp_sender is not None
                and subscriber.get("whatsapp_e164")
            ):
                key = build_dedupe_key(subscriber["user_id"], asset, ts)
                if _already_sent(db, key, "whatsapp"):
                    summary["skipped"] += 1
                elif dry_run:
                    summary["sent_whatsapp"] += 1
                else:
                    try:
                        _, params = format_whatsapp_params(asset, proba, ts)
                        msg_id = whatsapp_sender.send(
                            subscriber["whatsapp_e164"], whatsapp_template, params
                        )
                        _record(db, subscriber["user_id"], "whatsapp", asset, key, "sent", str(msg_id))
                        summary["sent_whatsapp"] += 1
                    except Exception:
                        logger.warning("whatsapp send failed for user %s", subscriber["user_id"], exc_info=True)
                        _record(db, subscriber["user_id"], "whatsapp", asset, key, "failed", None)
                        summary["failed"] += 1
    return summary
