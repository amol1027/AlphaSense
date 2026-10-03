"""Pure notification rules: no network, no DB, no prediction imports.

Decides whether a volatility-regime prediction warrants an email/WhatsApp
reminder for a given user's preferences.
"""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

KNOWN_ASSETS = ("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")
IST = ZoneInfo("Asia/Kolkata")

# Reminders are only sent during the normal NSE/BSE continuous session.
SESSION_START_MINUTES = 9 * 60 + 15  # 09:15 IST
SESSION_END_MINUTES = 15 * 60 + 30  # 15:30 IST (last alertable bar: 14:15 -> 15:15)

DEFAULT_MIN_PROBABILITY = 0.70
MIN_PROBABILITY_FLOOR = 0.50
MIN_PROBABILITY_CEILING = 0.95

_WHATSAPP_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")

DISCLAIMER = "Research note, not trading advice. Stop: profile/notifications."


def normalize_whatsapp(raw: str | None) -> str | None:
    """Strip spaces/dashes; return E.164 or None when blank."""
    if raw is None:
        return None
    cleaned = re.sub(r"[\s\-()]", "", raw.strip())
    if not cleaned:
        return None
    return cleaned


def is_valid_whatsapp(value: str | None) -> bool:
    return value is not None and _WHATSAPP_PATTERN.fullmatch(value) is not None


def normalize_assets(assets: list[str] | tuple[str, ...] | None) -> list[str]:
    if not assets:
        return list(KNOWN_ASSETS)
    out: list[str] = []
    for asset in assets:
        upper = str(asset).strip().upper()
        if upper in KNOWN_ASSETS and upper not in out:
            out.append(upper)
    return out or list(KNOWN_ASSETS)


def validate_prefs(prefs: dict) -> dict:
    """Normalize user prefs; raise ValueError on bad values."""
    email_enabled = bool(prefs.get("email_enabled", True))
    whatsapp_enabled = bool(prefs.get("whatsapp_enabled", False))
    assets = normalize_assets(prefs.get("assets"))
    try:
        min_probability = float(prefs.get("min_probability", DEFAULT_MIN_PROBABILITY))
    except (TypeError, ValueError):
        raise ValueError("min_probability must be a number") from None
    if not MIN_PROBABILITY_FLOOR <= min_probability <= MIN_PROBABILITY_CEILING:
        raise ValueError(f"min_probability must be between {MIN_PROBABILITY_FLOOR} and {MIN_PROBABILITY_CEILING}")
    whatsapp = normalize_whatsapp(prefs.get("whatsapp_e164"))
    if whatsapp is not None and not is_valid_whatsapp(whatsapp):
        raise ValueError("whatsapp_e164 must be E.164 like +919876543210")
    if whatsapp_enabled and not whatsapp:
        raise ValueError("A verified WhatsApp number is required to enable WhatsApp reminders")
    return {
        "email_enabled": email_enabled,
        "whatsapp_enabled": whatsapp_enabled,
        "whatsapp_e164": whatsapp,
        "assets": assets,
        "min_probability": min_probability,
        "daily_summary": bool(prefs.get("daily_summary", False)),
    }


def is_session_time(moment: datetime) -> bool:
    """True when moment falls inside 09:15-15:30 IST (Mon-Fri)."""
    ist_time = moment.astimezone(IST) if moment.tzinfo else moment.replace(tzinfo=IST)
    if ist_time.weekday() >= 5:
        return False
    minutes = ist_time.hour * 60 + ist_time.minute
    return SESSION_START_MINUTES <= minutes <= SESSION_END_MINUTES


def build_dedupe_key(user_id: int | str, asset: str, prediction_timestamp: str, kind: str = "high_range") -> str:
    return f"{user_id}:{str(asset).upper()}:{prediction_timestamp}:{kind}"


def should_notify(prediction: dict, prefs: dict, moment: datetime | None = None) -> tuple[bool, str]:
    """Return (notify, reason). Pure: safe to unit-test without DB."""
    try:
        clean = validate_prefs(prefs)
    except ValueError as exc:
        return False, f"invalid_prefs: {exc}"
    if not clean["email_enabled"] and not clean["whatsapp_enabled"]:
        return False, "all_channels_disabled"
    asset = str(prediction.get("asset", "")).upper()
    if asset not in clean["assets"]:
        return False, "asset_not_subscribed"
    try:
        proba = float(prediction.get("probability_high_range", prediction.get("probability", 0.0)))
    except (TypeError, ValueError):
        return False, "bad_probability"
    if prediction.get("prediction") != 1:
        return False, "not_high_range"
    if proba < clean["min_probability"]:
        return False, "below_threshold"
    if prediction.get("stale") is True:
        return False, "stale_prediction"
    if moment is not None and not is_session_time(moment):
        return False, "outside_session"
    return True, "ok"


def _ist_short(iso_timestamp: str) -> str:
    try:
        moment = datetime.fromisoformat(str(iso_timestamp).replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=IST)
        return moment.astimezone(IST).strftime("%H:%M IST")
    except (ValueError, TypeError):
        return str(iso_timestamp)


def format_email(asset: str, proba: float, prediction_timestamp: str) -> tuple[str, str]:
    subject = f"AlphaSense research: {asset.upper()} high-range regime {proba:.0%}"
    body = (
        f"AlphaSense research note (not trading advice).\n\n"
        f"{asset.upper()} shows a high next-hour range regime with p={proba:.2f} "
        f"at {_ist_short(prediction_timestamp)}.\n"
        f"Shadow volatility model, 1h horizon.\n\n{DISCLAIMER}"
    )
    return subject, body


def format_whatsapp_params(asset: str, proba: float, prediction_timestamp: str) -> tuple[str, list[str]]:
    """Return (template_name, params) for the Meta Cloud utility template."""
    return "volatility_alert", [asset.upper(), f"{proba:.0%}", _ist_short(prediction_timestamp)]
