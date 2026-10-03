"""Email + WhatsApp reminder runner (best-effort, never breaks refresh).

Usage:
  Dry run (no sends, no log writes for sends):  python scripts/send_reminders.py --dry-run
  Live run (NSE hours, every 15m via Task Scheduler):  python scripts/send_reminders.py
  Bypass session gate for testing:  python scripts/send_reminders.py --dry-run --force

Reads dashboard_snapshot() predictions, fans out via
src/notifications/dispatcher.py, and prints a JSON summary. Provider
failures are counted, never raised.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("alphasense.send_reminders")


def _build_senders(dry_run: bool):
    email_sender = None
    whatsapp_sender = None
    if dry_run:
        return None, None
    if os.getenv("SMTP_HOST"):
        from src.notifications.smtp_email import SmtpEmailSender

        try:
            email_sender = SmtpEmailSender.from_env()
        except ValueError as exc:
            logger.warning("SMTP disabled: %s", exc)
    if os.getenv("WHATSAPP_TOKEN") and os.getenv("WHATSAPP_PHONE_ID"):
        from src.notifications.whatsapp_cloud import WhatsAppCloudSender

        try:
            whatsapp_sender = WhatsAppCloudSender.from_env()
        except ValueError as exc:
            logger.warning("WhatsApp disabled: %s", exc)
    return email_sender, whatsapp_sender


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Send email/WhatsApp regime reminders")
    parser.add_argument("--dry-run", action="store_true", help="evaluate rules without sending")
    parser.add_argument("--force", action="store_true", help="bypass the IST session gate (testing)")
    args = parser.parse_args()

    from src.auth import database
    from src.notifications.dispatcher import send_for_snapshot
    from src.prediction.server import dashboard_snapshot

    try:
        snapshot = dashboard_snapshot()
    except Exception:
        logger.exception("Could not build dashboard snapshot")
        print(json.dumps({"error": "snapshot_failed"}))
        return 2
    predictions = snapshot.get("predictions", {})
    moment = None if args.force else datetime.now(timezone.utc)
    email_sender, whatsapp_sender = _build_senders(args.dry_run)
    if not args.dry_run and email_sender is None and whatsapp_sender is None:
        logger.warning("No providers configured (SMTP_HOST / WHATSAPP_TOKEN); nothing will send.")
    summary = send_for_snapshot(
        predictions,
        db=database,
        email_sender=email_sender,
        whatsapp_sender=whatsapp_sender,
        whatsapp_template=os.getenv("WHATSAPP_TEMPLATE", "volatility_alert"),
        moment=moment,
        dry_run=args.dry_run,
    )
    summary["dry_run"] = bool(args.dry_run)
    summary["feed"] = snapshot.get("feed", {}).get("state")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
