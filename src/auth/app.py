"""Cookie-based account API for the AlphaSense public research site."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError, VerificationError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from mysql.connector import IntegrityError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.auth import database


load_dotenv()

SESSION_COOKIE = "alpha_session"
CSRF_COOKIE = "alpha_csrf"
SESSION_DAYS = 7
PASSWORD_HASHER = PasswordHasher(time_cost=2, memory_cost=19_456, parallelism=1)
FAKE_HASH = PASSWORD_HASHER.hash("not-a-real-user-password")
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ALLOWED_ORIGINS = {
    value.strip().rstrip("/")
    for value in os.getenv(
        "AUTH_ALLOWED_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173",
    ).split(",")
    if value.strip()
}
SAME_SITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax").lower()
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}
if SAME_SITE not in {"lax", "strict", "none"}:
    raise RuntimeError("AUTH_COOKIE_SAMESITE must be lax, strict, or none")
if SAME_SITE == "none" and not COOKIE_SECURE:
    raise RuntimeError("AUTH_COOKIE_SECURE must be true when AUTH_COOKIE_SAMESITE=none")
if any(origin.startswith("https://") for origin in ALLOWED_ORIGINS) and not COOKIE_SECURE:
    raise RuntimeError("AUTH_COOKIE_SECURE must be true when serving HTTPS origins")


def _rate_limit_setting(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        raise RuntimeError(f"{name} must be an integer") from None
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


RATE_LIMIT = _rate_limit_setting("AUTH_RATE_LIMIT", 10)
RATE_WINDOW_SECONDS = _rate_limit_setting("AUTH_RATE_WINDOW_SECONDS", 600)

# Blocklist for the most reused weak passwords. Length is the real defense
# (see Registration); this only stops the laziest choices.
WEAK_PASSWORDS = frozenset({
    "password", "password1", "password123", "1234567890", "123456789",
    "qwerty123", "letmein123", "welcome123", "alphasense", "alphasense123",
})


class Credentials(BaseModel):
    """Login shape: any non-empty password reaches verification so that
    accounts created under an older policy can still sign in."""

    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not EMAIL_PATTERN.fullmatch(normalized):
            raise ValueError("Enter a valid email address")
        return normalized


class Registration(Credentials):
    password: str = Field(min_length=10, max_length=128)
    display_name: str = Field(min_length=1, max_length=80)
    # Reminder opt-ins collected at signup. Email defaults on (free);
    # WhatsApp defaults off and requires an E.164 number when enabled.
    email_updates: bool = True
    whatsapp_updates: bool = False
    whatsapp_e164: str | None = Field(default=None, max_length=20)

    @field_validator("password")
    @classmethod
    def reject_weak_password(cls, value: str) -> str:
        if value.strip().lower() in WEAK_PASSWORDS:
            raise ValueError("Choose a less common password")
        return value

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter your name")
        return value

    @field_validator("whatsapp_e164")
    @classmethod
    def normalize_whatsapp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from src.notifications.rules import is_valid_whatsapp, normalize_whatsapp

        cleaned = normalize_whatsapp(value)
        if cleaned is None:
            return None
        if not is_valid_whatsapp(cleaned):
            raise ValueError("Enter a valid WhatsApp number like +919876543210")
        return cleaned

    @model_validator(mode="after")
    def require_number_when_whatsapp_on(self):
        if self.whatsapp_updates and not self.whatsapp_e164:
            raise ValueError("Add your WhatsApp number to enable WhatsApp reminders")
        return self


class DisplayNameUpdate(BaseModel):
    """Self-service display-name edit (same normalization as registration)."""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=80)

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter your name")
        return value


class PasswordChange(BaseModel):
    """Password change: current password is verified; the new one must meet
    the registration policy (min length + blocklist)."""

    model_config = ConfigDict(extra="forbid")

    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=10, max_length=128)

    @field_validator("new_password")
    @classmethod
    def reject_weak_password(cls, value: str) -> str:
        if value.strip().lower() in WEAK_PASSWORDS:
            raise ValueError("Choose a less common password")
        return value


class SlidingWindowLimit:
    """Small per-process brake. Production deployments must run a single
    uvicorn worker or put a shared (edge/WAF) limiter in front; otherwise
    each worker enforces its own independent budget."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self.lock:
            events = self.events[key]
            while events and events[0] < now - self.window_seconds:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(events[0] + self.window_seconds - now))
                raise HTTPException(
                    status_code=429,
                    detail="Too many attempts. Wait a few minutes and try again.",
                    headers={"Retry-After": str(retry_after)},
                )
            events.append(now)


auth_limiter = SlidingWindowLimit(limit=RATE_LIMIT, window_seconds=RATE_WINDOW_SECONDS)
# Admin reads (table browsing/pagination) get their own budget so they can't
# starve login/register and vice versa. Writes still require CSRF + origin.
admin_limiter = SlidingWindowLimit(limit=60, window_seconds=60)


logger = logging.getLogger("alphasense.auth")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        database.check_schema()
    except Exception as exc:
        raise RuntimeError("Could not connect to the AlphaSense account tables. Check MySQL settings and run scripts/initialize_auth_db.py.") from exc
    logger.warning(
        "Rate limiter is per-process memory (limit=%s per %ss). "
        "Run a single uvicorn worker or enforce a shared edge limiter in production.",
        RATE_LIMIT,
        RATE_WINDOW_SECONDS,
    )
    yield


app = FastAPI(title="AlphaSense account API", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_same_origin(request: Request) -> None:
    origin = request.headers.get("origin", "").rstrip("/")
    if origin not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="Request origin is not allowed")


def _check_csrf(request: Request) -> None:
    _require_same_origin(request)
    cookie_token = request.cookies.get(CSRF_COOKIE, "")
    header_token = request.headers.get("x-csrf-token", "")
    if not cookie_token or not header_token or not hmac.compare_digest(cookie_token, header_token):
        raise HTTPException(status_code=403, detail="Refresh the page and try again")
    session_token = request.cookies.get(SESSION_COOKIE, "")
    if session_token:
        session = database.fetch_one(
            "SELECT csrf_hash FROM auth_sessions WHERE token_hash = %s AND expires_at > UTC_TIMESTAMP()",
            (_digest(session_token),),
        )
        if session and not hmac.compare_digest(session["csrf_hash"], _digest(cookie_token)):
            raise HTTPException(status_code=403, detail="Refresh the page and try again")


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _set_csrf_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        CSRF_COOKIE,
        token,
        httponly=False,
        secure=COOKIE_SECURE,
        samesite=SAME_SITE,
        path="/",
        max_age=SESSION_DAYS * 24 * 60 * 60,
    )


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=SAME_SITE,
        path="/",
        max_age=SESSION_DAYS * 24 * 60 * 60,
    )


def _new_session(user_id: int, response: Response) -> None:
    session_token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=SESSION_DAYS)
    database.execute(
        "INSERT INTO auth_sessions (token_hash, csrf_hash, user_id, expires_at) VALUES (%s, %s, %s, %s)",
        (_digest(session_token), _digest(csrf_token), user_id, expires_at),
    )
    _set_session_cookie(response, session_token)
    _set_csrf_cookie(response, csrf_token)


def _current_user(request: Request) -> dict | None:
    session_token = request.cookies.get(SESSION_COOKIE, "")
    if not session_token:
        return None
    user = database.fetch_one(
        """SELECT u.id, u.email, u.display_name, u.role, u.disabled_at
           FROM auth_sessions s JOIN users u ON u.id = s.user_id
           WHERE s.token_hash = %s AND s.expires_at > UTC_TIMESTAMP()
           AND u.disabled_at IS NULL""",
        (_digest(session_token),),
    )
    if not user:
        return None
    # Older DBs / test fakes without the M0 columns default to plain user.
    user.setdefault("role", "user")
    return user


def _require_admin(request: Request) -> dict:
    """M0 gate for future /api/admin/* routes: 401 anonymous, 403 non-admin."""
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    if user.get("disabled_at") is not None or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def _audit(actor_id: int, action: str, target_user_id: int | None = None, detail: str | None = None) -> None:
    """Best-effort audit write; never breaks the triggering request on old DBs."""
    try:
        database.execute(
            "INSERT INTO admin_audit_log (actor_id, action, target_user_id, detail) VALUES (%s, %s, %s, %s)",
            (actor_id, action, target_user_id, detail),
        )
    except Exception:
        logger.warning("admin audit write failed", exc_info=True)


@app.get("/api/auth/health")
def health() -> Response:
    """Liveness + MySQL reachability for supervisors. 503 on DB failure."""
    try:
        database.fetch_one("SELECT 1")
    except Exception:
        logger.exception("Auth health check failed")
        return Response(status_code=503)
    return Response(status_code=200)


@app.get("/api/auth/csrf")
def csrf_token(request: Request, response: Response) -> dict[str, str]:
    # Reuse the double-submit token when present. Rotating it on every GET can
    # desynchronize concurrent requests (React StrictMode issues duplicate
    # effects in development) and invalidate a session's CSRF hash.
    token = request.cookies.get(CSRF_COOKIE, "") or secrets.token_urlsafe(32)
    session_token = request.cookies.get(SESSION_COOKIE, "")
    if session_token:
        session = database.fetch_one(
            "SELECT csrf_hash FROM auth_sessions WHERE token_hash = %s AND expires_at > UTC_TIMESTAMP()",
            (_digest(session_token),),
        )
        if session and not hmac.compare_digest(session["csrf_hash"], _digest(token)):
            token = secrets.token_urlsafe(32)
            database.execute(
                "UPDATE auth_sessions SET csrf_hash = %s WHERE token_hash = %s AND expires_at > UTC_TIMESTAMP()",
                (_digest(token), _digest(session_token)),
            )
    _set_csrf_cookie(response, token)
    return {"csrfToken": token}


@app.post("/api/auth/register", status_code=201)
def register(payload: Registration, request: Request, response: Response) -> dict:
    _check_csrf(request)
    auth_limiter.check(_client_key(request))
    password_hash = PASSWORD_HASHER.hash(payload.password)
    try:
        user_id = database.execute(
            "INSERT INTO users (email, display_name, password_hash) VALUES (%s, %s, %s)",
            (payload.email, payload.display_name, password_hash),
        )
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Unable to create an account with those details") from exc
    _save_notification_signup(
        user_id,
        email_enabled=payload.email_updates,
        whatsapp_enabled=payload.whatsapp_updates,
        whatsapp_e164=payload.whatsapp_e164,
    )
    _new_session(user_id, response)
    _send_welcome_email(payload.email, payload.display_name)
    return {"id": user_id, "email": payload.email, "displayName": payload.display_name, "role": "user"}


@app.post("/api/auth/login")
def login(payload: Credentials, request: Request, response: Response) -> dict:
    _check_csrf(request)
    auth_limiter.check(_client_key(request))
    user = database.fetch_one(
        "SELECT id, email, display_name, password_hash, role, disabled_at FROM users WHERE email = %s",
        (payload.email,),
    )
    password_hash = user["password_hash"] if user else FAKE_HASH
    valid = False
    try:
        valid = PASSWORD_HASHER.verify(password_hash, payload.password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        valid = False
    if not user or not valid or user.get("disabled_at") is not None:
        raise HTTPException(status_code=401, detail="Email or password is incorrect")
    if PASSWORD_HASHER.check_needs_rehash(password_hash):
        database.execute("UPDATE users SET password_hash = %s WHERE id = %s", (PASSWORD_HASHER.hash(payload.password), user["id"]))
    _new_session(int(user["id"]), response)
    _send_login_alert_email(user["email"], user["display_name"], _client_key(request))
    return {"id": user["id"], "email": user["email"], "displayName": user["display_name"], "role": user.get("role", "user")}


@app.get("/api/auth/me")
def me(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    return {"id": user["id"], "email": user["email"], "displayName": user["display_name"], "role": user.get("role", "user")}


@app.patch("/api/auth/me")
def update_me(payload: DisplayNameUpdate, request: Request) -> dict:
    _check_csrf(request)
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    database.execute(
        "UPDATE users SET display_name = %s WHERE id = %s",
        (payload.display_name, user["id"]),
    )
    return {"id": user["id"], "email": user["email"], "displayName": payload.display_name, "role": user.get("role", "user")}


@app.post("/api/auth/me/password")
def change_password(payload: PasswordChange, request: Request) -> dict[str, bool]:
    _check_csrf(request)
    # Brake against current-password guessing; shares the login budget.
    auth_limiter.check(_client_key(request))
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=400, detail="The new password must be different")
    row = database.fetch_one(
        "SELECT id, email, display_name, password_hash, role, disabled_at FROM users WHERE id = %s",
        (user["id"],),
    )
    if not row or row.get("disabled_at") is not None:
        raise HTTPException(status_code=401, detail="Not signed in")
    try:
        current_valid = PASSWORD_HASHER.verify(row["password_hash"], payload.current_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        current_valid = False
    if not current_valid:
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    database.execute(
        "UPDATE users SET password_hash = %s WHERE id = %s",
        (PASSWORD_HASHER.hash(payload.new_password), user["id"]),
    )
    # Other sessions stay valid until they expire; use the admin revoke
    # surface or sign out everywhere if a compromise is suspected.
    return {"changed": True}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    _check_csrf(request)
    session_token = request.cookies.get(SESSION_COOKIE, "")
    if session_token:
        database.execute("DELETE FROM auth_sessions WHERE token_hash = %s", (_digest(session_token),))
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, secure=COOKIE_SECURE, samesite=SAME_SITE)
    csrf = secrets.token_urlsafe(32)
    _set_csrf_cookie(response, csrf)
    return {"signedOut": True}


# ---------------------------------------------------------------------------
# Notification preferences (email + WhatsApp reminders).
# Tables are created by migrations/002_notifications.sql; all writes are
# best-effort so older DBs without the migration keep working with defaults.
# ---------------------------------------------------------------------------

NOTIFICATION_ASSETS = ("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")


def _default_prefs() -> dict:
    return {
        "emailEnabled": True,
        "whatsappEnabled": False,
        "whatsappNumber": None,
        "assets": list(NOTIFICATION_ASSETS),
        "minProbability": 0.70,
        "dailySummary": False,
    }


def _save_notification_signup(
    user_id: int,
    *,
    email_enabled: bool,
    whatsapp_enabled: bool,
    whatsapp_e164: str | None,
) -> None:
    try:
        try:
            database.execute(
                "UPDATE users SET whatsapp_e164 = %s WHERE id = %s",
                (whatsapp_e164, user_id),
            )
        except Exception:
            pass  # older DBs without users.whatsapp_e164 keep working
        database.execute(
            "INSERT INTO notification_prefs (user_id, email_enabled, whatsapp_enabled, min_probability) "
            "VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
            "email_enabled = VALUES(email_enabled), whatsapp_enabled = VALUES(whatsapp_enabled)",
            (user_id, int(bool(email_enabled)), int(bool(whatsapp_enabled)), 0.700),
        )
    except Exception:
        logger.warning("notification signup prefs save failed", exc_info=True)


def _send_welcome_email(email: str, display_name: str) -> None:
    """Best-effort welcome email; never breaks registration."""
    try:
        from src.notifications.transactional import format_welcome_email, send_security_email

        subject, body = format_welcome_email(display_name)
        send_security_email(email, subject, body)
    except Exception:
        logger.warning("welcome email failed", exc_info=True)


def _send_login_alert_email(email: str, display_name: str, ip: str | None) -> None:
    """Best-effort sign-in alert; never breaks login."""
    try:
        from src.notifications.transactional import format_login_alert_email, send_security_email

        subject, body = format_login_alert_email(display_name, ip)
        send_security_email(email, subject, body)
    except Exception:
        logger.warning("login alert email failed", exc_info=True)


def _read_notification_prefs(user_id: int) -> dict:
    prefs = _default_prefs()
    whatsapp: str | None = None
    try:
        row = database.fetch_one(
            "SELECT email_enabled, whatsapp_enabled, assets, min_probability, daily_summary "
            "FROM notification_prefs WHERE user_id = %s",
            (user_id,),
        )
    except Exception:
        row = None
    try:
        user_row = database.fetch_one("SELECT whatsapp_e164 FROM users WHERE id = %s", (user_id,))
        if user_row:
            whatsapp = user_row.get("whatsapp_e164")
    except Exception:
        whatsapp = None
    if row:
        import json as _prefs_json

        prefs["emailEnabled"] = bool(row.get("email_enabled", 1))
        prefs["whatsappEnabled"] = bool(row.get("whatsapp_enabled", 0))
        raw_assets = row.get("assets")
        if raw_assets:
            try:
                parsed = _prefs_json.loads(raw_assets) if isinstance(raw_assets, str) else list(raw_assets)
                cleaned = [a for a in (str(x).upper() for x in parsed) if a in NOTIFICATION_ASSETS]
                if cleaned:
                    prefs["assets"] = cleaned
            except (ValueError, TypeError):
                pass
        try:
            prefs["minProbability"] = float(row.get("min_probability", 0.70))
        except (TypeError, ValueError):
            pass
        prefs["dailySummary"] = bool(row.get("daily_summary", 0))
    prefs["whatsappNumber"] = whatsapp
    return prefs


class NotificationPrefsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email_enabled: bool | None = None
    whatsapp_enabled: bool | None = None
    whatsapp_e164: str | None = Field(default=None, max_length=20)
    assets: list[str] | None = None
    min_probability: float | None = Field(default=None, ge=0.5, le=0.95)
    daily_summary: bool | None = None

    @field_validator("whatsapp_e164")
    @classmethod
    def normalize_whatsapp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from src.notifications.rules import is_valid_whatsapp, normalize_whatsapp

        cleaned = normalize_whatsapp(value)
        if cleaned is None:
            return None  # clearing the number
        if not is_valid_whatsapp(cleaned):
            raise ValueError("Enter a valid WhatsApp number like +919876543210")
        return cleaned

    @field_validator("assets")
    @classmethod
    def normalize_assets(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [str(a).strip().upper() for a in value if str(a).strip()]
        unknown = [a for a in cleaned if a not in NOTIFICATION_ASSETS]
        if unknown:
            raise ValueError(f"Unknown assets: {', '.join(unknown)}")
        if not cleaned:
            raise ValueError("Choose at least one asset")
        return sorted(set(cleaned), key=cleaned.index)


@app.get("/api/auth/me/notifications")
def get_notification_prefs(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    return _read_notification_prefs(int(user["id"]))


@app.patch("/api/auth/me/notifications")
def update_notification_prefs(payload: NotificationPrefsUpdate, request: Request) -> dict:
    _check_csrf(request)
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    user_id = int(user["id"])
    current = _read_notification_prefs(user_id)
    email_enabled = payload.email_enabled if payload.email_enabled is not None else current["emailEnabled"]
    whatsapp_enabled = (
        payload.whatsapp_enabled if payload.whatsapp_enabled is not None else current["whatsappEnabled"]
    )
    whatsapp_number = payload.whatsapp_e164 if "whatsapp_e164" in payload.model_fields_set else current["whatsappNumber"]
    if whatsapp_enabled and not whatsapp_number:
        raise HTTPException(status_code=400, detail="Add your WhatsApp number to enable WhatsApp reminders")
    assets = payload.assets if payload.assets is not None else current["assets"]
    min_probability = payload.min_probability if payload.min_probability is not None else current["minProbability"]
    daily_summary = payload.daily_summary if payload.daily_summary is not None else current["dailySummary"]
    try:
        try:
            database.execute("UPDATE users SET whatsapp_e164 = %s WHERE id = %s", (whatsapp_number, user_id))
        except Exception:
            pass  # older DBs without the column
        database.execute(
            "INSERT INTO notification_prefs (user_id, email_enabled, whatsapp_enabled, assets, min_probability, daily_summary) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
            "email_enabled = VALUES(email_enabled), whatsapp_enabled = VALUES(whatsapp_enabled), "
            "assets = VALUES(assets), min_probability = VALUES(min_probability), daily_summary = VALUES(daily_summary)",
            (user_id, int(bool(email_enabled)), int(bool(whatsapp_enabled)), _json.dumps(assets), float(min_probability), int(bool(daily_summary))),
        )
    except Exception:
        logger.warning("notification prefs save failed", exc_info=True)
        raise HTTPException(status_code=503, detail="Reminder preferences are unavailable right now")
    return _read_notification_prefs(user_id)


# ---------------------------------------------------------------------------
# Admin API (M1): localhost-operator surface, proxied only through this service.
# Prediction :8000 stays unauthenticated locally; cross-host admin must use a
# reverse proxy, not direct :8000 exposure. Every mutation is audited.
# ---------------------------------------------------------------------------

import json as _json


class AdminUserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    role: str | None = Field(default=None, pattern="^(user|admin)$")
    disabled: bool | None = None

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value:
            raise ValueError("Enter a name")
        return value


def _public_user(row: dict) -> dict:
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "displayName": row["display_name"],
        "role": row.get("role", "user"),
        "disabled": row.get("disabled_at") is not None,
        "disabledAt": row["disabled_at"].isoformat() if row.get("disabled_at") else None,
        "createdAt": row["created_at"].isoformat() if row.get("created_at") else None,
    }


def _clamp_page(limit: int, offset: int, max_limit: int = 100) -> tuple[int, int]:
    return (max(1, min(limit, max_limit)), max(0, offset))


def _admin_check_read(request: Request) -> dict:
    admin = _require_admin(request)
    admin_limiter.check(f"admin:{_client_key(request)}")
    return admin


@app.get("/api/admin/users")
def admin_list_users(request: Request, search: str = "", limit: int = 25, offset: int = 0, include_disabled: bool = False) -> dict:
    _admin_check_read(request)
    limit, offset = _clamp_page(limit, offset)
    clauses = []
    params: list = []
    if not include_disabled:
        clauses.append("disabled_at IS NULL")
    if search.strip():
        clauses.append("(email LIKE %s OR display_name LIKE %s)")
        like = f"%{search.strip()}%"
        params.extend([like, like])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    total_row = database.fetch_one(f"SELECT COUNT(*) AS total FROM users {where}", tuple(params))
    rows = database.fetch_all(
        f"SELECT id, email, display_name, role, disabled_at, created_at FROM users {where} "
        "ORDER BY id ASC LIMIT %s OFFSET %s",
        tuple(params) + (limit, offset),
    )
    return {"users": [_public_user(r) for r in rows], "total": int((total_row or {}).get("total", 0)), "limit": limit, "offset": offset}


@app.get("/api/admin/users/{user_id}")
def admin_get_user(user_id: int, request: Request) -> dict:
    _admin_check_read(request)
    row = database.fetch_one(
        "SELECT id, email, display_name, role, disabled_at, created_at FROM users WHERE id = %s",
        (user_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return _public_user(row)


@app.patch("/api/admin/users/{user_id}")
def admin_update_user(user_id: int, payload: AdminUserUpdate, request: Request) -> dict:
    _check_csrf(request)
    admin = _require_admin(request)
    row = database.fetch_one(
        "SELECT id, email, display_name, role, disabled_at FROM users WHERE id = %s",
        (user_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    current_role = row.get("role", "user")
    currently_disabled = row.get("disabled_at") is not None
    changes: dict = {}

    new_name = payload.display_name if payload.display_name is not None else row["display_name"]
    if new_name != row["display_name"]:
        changes["display_name"] = new_name
    new_role = payload.role if payload.role is not None else current_role
    if new_role != current_role:
        if int(row["id"]) == int(admin["id"]):
            raise HTTPException(status_code=400, detail="You cannot change your own role")
        changes["role"] = new_role
    new_disabled = currently_disabled if payload.disabled is None else bool(payload.disabled)
    if new_disabled != currently_disabled:
        if int(row["id"]) == int(admin["id"]):
            raise HTTPException(status_code=400, detail="You cannot disable your own account")
        changes["disabled"] = new_disabled

    if not changes:
        return _public_user({**row, "display_name": new_name})

    # Last-admin guard: demoting or disabling the final active admin is refused.
    if (changes.get("role") == "user" and current_role == "admin" and not currently_disabled) or (
        changes.get("disabled") is True and current_role == "admin"
    ):
        count_row = database.fetch_one(
            "SELECT COUNT(*) AS total FROM users WHERE role = 'admin' AND disabled_at IS NULL", ()
        )
        if int((count_row or {}).get("total", 0)) <= 1:
            raise HTTPException(status_code=409, detail="Refusing: this is the last active admin")

    if "display_name" in changes:
        database.execute("UPDATE users SET display_name = %s WHERE id = %s", (changes["display_name"], user_id))
    if "role" in changes:
        database.execute("UPDATE users SET role = %s WHERE id = %s", (changes["role"], user_id))
    if changes.get("disabled") is True:
        database.execute("UPDATE users SET disabled_at = UTC_TIMESTAMP() WHERE id = %s", (user_id,))
        # Sessions left in the table are inert: _current_user filters disabled
        # accounts, but delete them anyway so the session list stays truthful.
        database.execute("DELETE FROM auth_sessions WHERE user_id = %s", (user_id,))
    elif changes.get("disabled") is False:
        database.execute("UPDATE users SET disabled_at = NULL WHERE id = %s", (user_id,))
    _audit(int(admin["id"]), "user.update", user_id, _json.dumps(changes, default=str))
    updated = database.fetch_one(
        "SELECT id, email, display_name, role, disabled_at, created_at FROM users WHERE id = %s",
        (user_id,),
    )
    return _public_user(updated or row)


@app.post("/api/admin/users/{user_id}/revoke-sessions")
def admin_revoke_user_sessions(user_id: int, request: Request) -> dict:
    _check_csrf(request)
    admin = _require_admin(request)
    if int(user_id) == int(admin["id"]):
        raise HTTPException(status_code=400, detail="Use sign out for your own sessions")
    row = database.fetch_one("SELECT id FROM users WHERE id = %s", (user_id,))
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    database.execute("DELETE FROM auth_sessions WHERE user_id = %s", (user_id,))
    _audit(int(admin["id"]), "session.revoke_all", user_id, None)
    return {"revoked": True}


@app.get("/api/admin/sessions")
def admin_list_sessions(request: Request, user_id: int | None = None, limit: int = 25) -> dict:
    _admin_check_read(request)
    limit, _ = _clamp_page(limit, 0)
    if user_id is not None:
        rows = database.fetch_all(
            "SELECT token_hash, user_id, expires_at, created_at FROM auth_sessions "
            "WHERE user_id = %s AND expires_at > UTC_TIMESTAMP() ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
    else:
        rows = database.fetch_all(
            "SELECT token_hash, user_id, expires_at, created_at FROM auth_sessions "
            "WHERE expires_at > UTC_TIMESTAMP() ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
    items = [
        {
            "tokenHash": r["token_hash"],
            "userId": int(r["user_id"]),
            "expiresAt": r["expires_at"].isoformat() if r.get("expires_at") else None,
            "createdAt": r["created_at"].isoformat() if r.get("created_at") else None,
        }
        for r in rows
    ]
    return {"sessions": items, "limit": limit}


@app.delete("/api/admin/sessions/{token_hash}")
def admin_revoke_session(token_hash: str, request: Request) -> dict:
    _check_csrf(request)
    admin = _require_admin(request)
    if not re.fullmatch(r"[0-9a-f]{64}", token_hash or ""):
        raise HTTPException(status_code=400, detail="Invalid session reference")
    if _digest(request.cookies.get(SESSION_COOKIE, "")) == token_hash:
        raise HTTPException(status_code=400, detail="Use sign out for your own session")
    database.execute("DELETE FROM auth_sessions WHERE token_hash = %s", (token_hash,))
    _audit(int(admin["id"]), "session.revoke_one", None, token_hash[:12])
    return {"revoked": True}


@app.get("/api/admin/audit")
def admin_list_audit(
    request: Request, action: str = "", actor_id: int | None = None,
    target_user_id: int | None = None, limit: int = 25, offset: int = 0,
) -> dict:
    _admin_check_read(request)
    limit, offset = _clamp_page(limit, offset)
    clauses = []
    params: list = []
    if action.strip():
        clauses.append("action = %s")
        params.append(action.strip())
    if actor_id is not None:
        clauses.append("actor_id = %s")
        params.append(actor_id)
    if target_user_id is not None:
        clauses.append("target_user_id = %s")
        params.append(target_user_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = database.fetch_all(
        f"SELECT id, actor_id, action, target_user_id, detail, created_at FROM admin_audit_log "
        f"{where} ORDER BY id DESC LIMIT %s OFFSET %s",
        tuple(params) + (limit, offset),
    )
    items = [
        {
            "id": int(r["id"]), "actorId": int(r["actor_id"]), "action": r["action"],
            "targetUserId": int(r["target_user_id"]) if r.get("target_user_id") is not None else None,
            "detail": r.get("detail"),
            "createdAt": r["created_at"].isoformat() if r.get("created_at") else None,
        }
        for r in rows
    ]
    return {"entries": items, "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Admin ops (M2): read-mostly system visibility + gated refresh, all served
# through this service. Browsers never call the prediction API directly from
# the admin UI; this service probes 127.0.0.1:8000 server-side. The prediction
# API stays localhost-only with no auth of its own.
# ---------------------------------------------------------------------------

import csv as _csv
import urllib.error as _urllib_error
import urllib.request as _urllib_request
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
_MARKET_CACHE_PATH = _PROJECT_ROOT / "data/interim/live_market_15m.csv"
_LIVE_STATUS_PATH = _PROJECT_ROOT / "data/interim/live_status.json"
_LIVE_LOG_PATH = _PROJECT_ROOT / "data/interim/live_log.csv"
_NEWS_STATUS_PATH = _PROJECT_ROOT / "data/interim/live_news_status.json"
_ARTIFACT_DIR = _PROJECT_ROOT / "artifacts/volatility"
_WEBAPP_INDEX = _PROJECT_ROOT / "website" / "dist" / "index.html"
OPS_KNOWN_ASSETS = ("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")
PREDICTION_HEALTH_URL = os.getenv("PREDICTION_HEALTH_URL", "http://127.0.0.1:8000/health")
_REFRESH_LOCK = threading.Lock()


def _read_json_file(path: _Path) -> dict | None:
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _prediction_health(timeout_seconds: int = 4) -> dict:
    """Server-side probe of the local prediction API. Never raises, never
    leaks anything beyond what /health already returns."""
    try:
        with _urllib_request.urlopen(PREDICTION_HEALTH_URL, timeout=timeout_seconds) as response:
            status = int(response.status)
            try:
                body = _json.loads(response.read().decode("utf-8"))
            except ValueError:
                body = None
    except _urllib_error.HTTPError as exc:
        try:
            body = _json.loads(exc.read().decode("utf-8"))
        except (ValueError, OSError):
            body = None
        return {"ok": False, "httpStatus": int(exc.code), "body": body, "error": None}
    except Exception as exc:
        logger.warning("prediction health probe failed: %s", exc)
        return {"ok": False, "httpStatus": None, "body": None, "error": "prediction API is unreachable"}
    checks = body.get("checks", {}) if isinstance(body, dict) else {}
    feed = body.get("feed", {}) if isinstance(body, dict) else {}
    return {
        "ok": status == 200,
        "httpStatus": status,
        "status": (body or {}).get("status") if isinstance(body, dict) else None,
        "feedState": feed.get("state"),
        "feedAgeMinutes": feed.get("age_minutes"),
        "checks": {name: bool(check.get("ok")) for name, check in checks.items()} if isinstance(checks, dict) else {},
        "error": None,
    }


def _csv_info(path: _Path) -> dict:
    try:
        stat = path.stat()
    except OSError:
        return {"present": False, "rows": None, "mtime": None}
    rows = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for _ in handle:
                rows += 1
        rows = max(0, rows - 1)  # header
    except OSError:
        return {"present": True, "rows": None, "mtime": None}
    return {
        "present": True,
        "rows": rows,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
    }


def _artifact_presence() -> dict:
    present = []
    missing = []
    for asset in OPS_KNOWN_ASSETS:
        if (_ARTIFACT_DIR / f"{asset.lower()}.joblib").is_file():
            present.append(asset)
        else:
            missing.append(asset)
    return {"ok": not missing, "present": present, "missing": missing}


def _system_status() -> dict:
    try:
        database.fetch_one("SELECT 1")
        auth_db_ok = True
    except Exception:
        auth_db_ok = False
    prediction = _prediction_health()
    cache = _csv_info(_MARKET_CACHE_PATH)
    cache["refresh"] = _read_json_file(_LIVE_STATUS_PATH)
    shadow = _csv_info(_LIVE_LOG_PATH)
    return {
        "authDb": {"ok": auth_db_ok},
        "prediction": prediction,
        "marketCache": cache,
        "newsCache": {
            **_csv_info(_NEWS_STATUS_PATH.parent / "live_news.csv"),
            "refresh": _read_json_file(_NEWS_STATUS_PATH),
        },
        "shadowLog": shadow,
        "modelArtifacts": _artifact_presence(),
        # Presence only — the value never leaves the server environment.
        "upstoxTokenConfigured": {"ok": bool(os.getenv("UPSTOX_ACCESS_TOKEN"))},
        "webapp": {"ok": _WEBAPP_INDEX.is_file()},
    }


def _read_shadow_log() -> list[dict]:
    """Raw live_log.csv rows (oldest first). Empty list when no log exists."""
    try:
        handle = _LIVE_LOG_PATH.open("r", encoding="utf-8", newline="")
    except OSError:
        return []
    with handle:
        return list(_csv.DictReader(handle))


def _shadow_tail(limit: int = 50) -> dict:
    limit = max(1, min(limit, 200))
    all_rows = _read_shadow_log()
    if not all_rows and not _LIVE_LOG_PATH.exists():
        return {"available": False, "rows": [], "counts": {"scored": 0, "correct": 0, "pending": 0}, "limit": limit}
    tail = all_rows[-limit:]
    rows = [
        {
            "asset": row.get("asset"),
            "predictionTimestamp": row.get("prediction_timestamp"),
            "probability": _to_float_or_none(row.get("probability")),
            "prediction": _to_int_or_none(row.get("prediction")),
            "realizedLabel": _to_int_or_none(row.get("realized_label")),
            "correct": _to_bool_or_none(row.get("correct")),
        }
        for row in tail
    ]
    scored = [r for r in rows if r["correct"] is not None]
    return {
        "available": True,
        "rows": rows,
        "counts": {
            "scored": len(scored),
            "correct": sum(1 for r in scored if r["correct"]),
            "pending": len(rows) - len(scored),
        },
        "limit": limit,
    }


def _to_float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value: str | None) -> int | None:
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _to_bool_or_none(value: str | None) -> bool | None:
    parsed = _to_int_or_none(value)
    return None if parsed is None else bool(parsed)


def _system_config() -> dict:
    return {
        "assets": list(OPS_KNOWN_ASSETS),
        "horizon": "1h",
        "session": "09:15-15:30 IST",
        "timezone": "Asia/Kolkata",
        "marketCacheRetentionDays": 7,
        "model": "frozen 9-feature 1h volatility regime (train via scripts/train_volatility_model.py; no UI retrain)",
        "predictionHealthUrl": "http://127.0.0.1:8000/health" if "127.0.0.1" in PREDICTION_HEALTH_URL else "custom",
        "allowedOrigins": sorted(ALLOWED_ORIGINS),
        "cookieSameSite": SAME_SITE,
        "cookieSecure": COOKIE_SECURE,
    }


def _run_system_refresh() -> dict:
    """Same flow as prediction POST /refresh: candles + shadow label + log.
    Separated for testability; the endpoint adds locking, audit, and errors."""
    from scripts.log_live_prediction import cmd_label as label_shadow_outcomes
    from scripts.log_live_prediction import cmd_log as log_shadow_predictions
    from scripts.refresh_live_market import refresh as refresh_market_data

    assets_updated = refresh_market_data()
    outcomes_scored = label_shadow_outcomes()
    predictions_added = log_shadow_predictions()
    return {
        "assetsUpdated": int(assets_updated),
        "outcomesScored": int(outcomes_scored),
        "predictionsAdded": int(predictions_added),
    }


@app.get("/api/admin/system/status")
def admin_system_status(request: Request) -> dict:
    _admin_check_read(request)
    return _system_status()


@app.get("/api/admin/system/shadow")
def admin_system_shadow(request: Request, limit: int = 50) -> dict:
    _admin_check_read(request)
    return _shadow_tail(limit)


@app.get("/api/admin/system/config")
def admin_system_config(request: Request) -> dict:
    _admin_check_read(request)
    return _system_config()


@app.post("/api/admin/system/refresh")
def admin_system_refresh(request: Request) -> dict:
    _check_csrf(request)
    admin = _require_admin(request)
    if not _REFRESH_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A market refresh is already in progress")
    try:
        try:
            result = _run_system_refresh()
        except Exception:
            logger.exception("Admin system refresh failed")
            raise HTTPException(status_code=502, detail="Market refresh failed. Check the server log and try again.")
        _audit(int(admin["id"]), "system.refresh", None, _json.dumps(result, default=str))
        return result
    finally:
        _REFRESH_LOCK.release()


# ---------------------------------------------------------------------------
# Admin analytics + exports: aggregated account/shadow stats and CSV downloads.
# Day buckets use Asia/Kolkata calendar days (naive DB datetimes are treated
# as UTC; exact day attribution is low-stakes for admin stats). Row counts are
# capped so a large table cannot exhaust the single auth worker.
# ---------------------------------------------------------------------------

import io as _io
from zoneinfo import ZoneInfo as _ZoneInfo

_ANALYTICS_DAYS = 30
_ANALYTICS_ROW_CAP = 5000
_IST = _ZoneInfo("Asia/Kolkata")


def _ist_day_key(value) -> str | None:
    if value is None:
        return None
    try:
        moment = value if getattr(value, "tzinfo", None) else value.replace(tzinfo=UTC)
        if isinstance(moment, str):
            moment = datetime.fromisoformat(moment)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(_IST).date().isoformat()
    except (TypeError, ValueError, AttributeError):
        return None


def _ist_day_series(days: int = _ANALYTICS_DAYS) -> list[str]:
    today = datetime.now(UTC).astimezone(_IST).date()
    return [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]


def _audit_family(action: str) -> str:
    if action.startswith("user."):
        return "user"
    if action.startswith("session."):
        return "session"
    if action.startswith("system."):
        return "system"
    return "other"


def _analytics_overview() -> dict:
    days = _ist_day_series()

    user_rows = database.fetch_all(
        "SELECT id, email, display_name, role, disabled_at, created_at FROM users "
        "ORDER BY id ASC LIMIT %s",
        (_ANALYTICS_ROW_CAP,),
    )
    session_rows = database.fetch_all(
        "SELECT user_id, created_at, expires_at FROM auth_sessions "
        "ORDER BY created_at DESC LIMIT %s",
        (_ANALYTICS_ROW_CAP,),
    )
    audit_rows = database.fetch_all(
        "SELECT action, actor_id, target_user_id, created_at FROM admin_audit_log "
        "ORDER BY id DESC LIMIT %s",
        (_ANALYTICS_ROW_CAP,),
    )

    users_total = len(user_rows)
    users_disabled = sum(1 for u in user_rows if u.get("disabled_at") is not None)
    users_admins = sum(1 for u in user_rows if u.get("role") == "admin" and u.get("disabled_at") is None)
    signups = {day: 0 for day in days}
    for user in user_rows:
        key = _ist_day_key(user.get("created_at"))
        if key in signups:
            signups[key] += 1

    now_naive = datetime.now(UTC).replace(tzinfo=None)
    live_sessions = [
        s for s in session_rows
        if s.get("expires_at") is not None and s["expires_at"] > now_naive
    ]
    sessions_per_day = {day: 0 for day in days}
    for session in session_rows:
        key = _ist_day_key(session.get("created_at"))
        if key in sessions_per_day:
            sessions_per_day[key] += 1

    email_by_id = {int(u["id"]): u["email"] for u in user_rows if u.get("email")}
    audit_per_day = {day: {"user": 0, "session": 0, "system": 0, "other": 0} for day in days}
    by_actor: dict[int, int] = {}
    last_24h = 0
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    for entry in audit_rows:
        created = entry.get("created_at")
        if isinstance(created, datetime):
            aware = created if created.tzinfo else created.replace(tzinfo=UTC)
            if aware >= cutoff:
                last_24h += 1
        key = _ist_day_key(created)
        if key in audit_per_day:
            audit_per_day[key][_audit_family(str(entry.get("action", "")))] += 1
        actor = entry.get("actor_id")
        if actor is not None:
            by_actor[int(actor)] = by_actor.get(int(actor), 0) + 1
    top_actors = sorted(by_actor.items(), key=lambda item: item[1], reverse=True)[:5]

    shadow_rows = _read_shadow_log()
    parsed = []
    for row in shadow_rows:
        try:
            moment = datetime.fromisoformat(str(row.get("prediction_timestamp", "")))
        except (TypeError, ValueError):
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        parsed.append({
            "day": moment.astimezone(_IST).date().isoformat(),
            "asset": row.get("asset"),
            "probability": _to_float_or_none(row.get("probability")),
            "correct": _to_bool_or_none(row.get("correct")),
        })
    logged_per_day = {day: 0 for day in days}
    scored_per_day: dict[str, list[int]] = {day: [] for day in days}
    for item in parsed:
        if item["day"] in logged_per_day:
            logged_per_day[item["day"]] += 1
            if item["correct"] is not None:
                scored_per_day[item["day"]].append(1 if item["correct"] else 0)
    per_day_shadow = [
        {
            "date": day,
            "logged": logged_per_day[day],
            "scored": len(scored_per_day[day]),
            "correct": sum(scored_per_day[day]),
        }
        for day in days
    ]
    rolling_7d = []
    for pos, day in enumerate(days):
        window = days[max(0, pos - 6):pos + 1]
        results = [v for d in window for v in scored_per_day[d]]
        rolling_7d.append({
            "date": day,
            "accuracy": round(sum(results) / len(results), 4) if results else None,
        })
    per_asset: dict[str, dict] = {}
    for item in parsed:
        asset = str(item["asset"] or "UNKNOWN")
        bucket = per_asset.setdefault(asset, {"logged": 0, "scored": 0, "correct": 0})
        bucket["logged"] += 1
        if item["correct"] is not None:
            bucket["scored"] += 1
            bucket["correct"] += 1 if item["correct"] else 0
    confidence = [0] * 10
    for item in parsed:
        prob = item["probability"]
        if prob is not None:
            confidence[max(0, min(9, int(prob * 10)))] += 1
    scored_all = [1 if i["correct"] else 0 for i in parsed if i["correct"] is not None]

    return {
        "users": {
            "total": users_total,
            "active": users_total - users_disabled,
            "disabled": users_disabled,
            "admins": users_admins,
            "signupsPerDay": [{"date": day, "count": signups[day]} for day in days],
        },
        "sessions": {
            "live": len(live_sessions),
            "perDay": [{"date": day, "count": sessions_per_day[day]} for day in days],
            "avgPerUser": round(len(session_rows) / users_total, 2) if users_total else 0,
        },
        "audit": {
            "last24h": last_24h,
            "perDay": [{"date": day, **audit_per_day[day]} for day in days],
            "topActors": [
                {"actorId": actor, "email": email_by_id.get(actor), "count": count}
                for actor, count in top_actors
            ],
        },
        "shadow": {
            "total": len(parsed),
            "scored": len(scored_all),
            "correct": sum(scored_all),
            "pending": len(parsed) - len(scored_all),
            "accuracy": round(sum(scored_all) / len(scored_all), 4) if scored_all else None,
            "perDay": per_day_shadow,
            "rolling7d": rolling_7d,
            "perAsset": [
                {
                    "asset": asset,
                    "logged": stats["logged"],
                    "scored": stats["scored"],
                    "correct": stats["correct"],
                    "accuracy": round(stats["correct"] / stats["scored"], 4) if stats["scored"] else None,
                }
                for asset, stats in sorted(per_asset.items())
            ],
            "confidence": confidence,
        },
    }


def _csv_cell(value) -> str:
    text = "" if value is None else str(value)
    if text[:1] in ("=", "+", "-", "@"):
        return "'" + text
    return text


def _csv_response(filename: str, header: list[str], rows: list[list]) -> Response:
    buffer = _io.StringIO()
    writer = _csv.writer(buffer)
    writer.writerow(header)
    for row in rows[:_ANALYTICS_ROW_CAP]:
        writer.writerow([_csv_cell(cell) for cell in row])
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/admin/analytics/overview")
def admin_analytics_overview(request: Request) -> dict:
    _admin_check_read(request)
    return _analytics_overview()


@app.get("/api/admin/export/users")
def admin_export_users(request: Request) -> Response:
    _admin_check_read(request)
    rows = database.fetch_all(
        "SELECT id, email, display_name, role, disabled_at, created_at FROM users "
        "ORDER BY id ASC LIMIT %s",
        (_ANALYTICS_ROW_CAP,),
    )
    return _csv_response(
        "alphasense-users.csv",
        ["id", "email", "display_name", "role", "disabled", "created_at"],
        [[r["id"], r.get("email"), r.get("display_name"), r.get("role", "user"),
          r.get("disabled_at") is not None, r.get("created_at")] for r in rows],
    )


@app.get("/api/admin/export/audit")
def admin_export_audit(request: Request) -> Response:
    _admin_check_read(request)
    rows = database.fetch_all(
        "SELECT id, actor_id, action, target_user_id, detail, created_at FROM admin_audit_log "
        "ORDER BY id DESC LIMIT %s",
        (_ANALYTICS_ROW_CAP,),
    )
    return _csv_response(
        "alphasense-audit.csv",
        ["id", "actor_id", "action", "target_user_id", "detail", "created_at"],
        [[r["id"], r.get("actor_id"), r.get("action"), r.get("target_user_id"),
          r.get("detail"), r.get("created_at")] for r in rows],
    )


@app.get("/api/admin/export/shadow")
def admin_export_shadow(request: Request) -> Response:
    _admin_check_read(request)
    if not _LIVE_LOG_PATH.exists():
        return _csv_response("alphasense-shadow.csv", ["asset", "prediction_timestamp"], [])
    return Response(
        content=_LIVE_LOG_PATH.read_bytes(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="alphasense-shadow.csv"'},
    )
