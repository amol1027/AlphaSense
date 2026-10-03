"""Email + WhatsApp notification layer (separate from prediction logic)."""

from src.notifications.dispatcher import get_subscribers, send_for_snapshot
from src.notifications.protocols import EmailSender, WhatsAppSender
from src.notifications.rules import (
    KNOWN_ASSETS,
    build_dedupe_key,
    format_email,
    format_whatsapp_params,
    normalize_assets,
    normalize_whatsapp,
    should_notify,
    validate_prefs,
)
from src.notifications.smtp_email import SmtpEmailSender
from src.notifications.transactional import (
    format_login_alert_email,
    format_welcome_email,
    send_security_email,
)
from src.notifications.whatsapp_cloud import WhatsAppCloudSender

__all__ = [
    "EmailSender",
    "WhatsAppSender",
    "SmtpEmailSender",
    "WhatsAppCloudSender",
    "get_subscribers",
    "send_for_snapshot",
    "KNOWN_ASSETS",
    "build_dedupe_key",
    "format_email",
    "format_login_alert_email",
    "format_welcome_email",
    "format_whatsapp_params",
    "normalize_assets",
    "normalize_whatsapp",
    "send_security_email",
    "should_notify",
    "validate_prefs",
]
