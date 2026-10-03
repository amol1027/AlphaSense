"""Notification rules + prefs API contract tests (no MySQL required)."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.auth import app as auth_app
from src.auth import database
from src.notifications import rules
from tests.test_auth_api import FakeDB, _csrf_headers, _register


@pytest.fixture
def notif_client(monkeypatch):
    fake = FakeDB()
    # notification tables live alongside users/sessions in the fake
    fake.prefs = {}
    fake.notif_log = []

    orig_fetch = fake.fetch_one
    orig_execute = fake.execute

    def fetch_one(query: str, params: tuple = ()):
        flat = " ".join(query.split())
        if flat.startswith("SELECT email_enabled, whatsapp_enabled"):
            row = fake.prefs.get(params[0])
            return dict(row) if row else None
        if flat.startswith("SELECT whatsapp_e164 FROM users"):
            user = next((u for u in fake.users.values() if u["id"] == params[0]), None)
            return {"whatsapp_e164": (user or {}).get("whatsapp_e164")} if user else None
        return orig_fetch(query, params)

    def execute(query: str, params: tuple = ()):
        flat = " ".join(query.split())
        if flat.startswith("UPDATE users SET whatsapp_e164"):
            for user in fake.users.values():
                if user["id"] == params[1]:
                    user["whatsapp_e164"] = params[0]
            return 0
        if flat.startswith("INSERT INTO notification_prefs"):
            import json as _json

            if len(params) == 4:  # signup shape
                user_id, email_enabled, whatsapp_enabled, min_prob = params
                fake.prefs[user_id] = {
                    "email_enabled": email_enabled,
                    "whatsapp_enabled": whatsapp_enabled,
                    "assets": None,
                    "min_probability": min_prob,
                    "daily_summary": 0,
                }
            else:  # PATCH shape
                user_id, email_enabled, whatsapp_enabled, assets, min_prob, daily = params
                fake.prefs[user_id] = {
                    "email_enabled": email_enabled,
                    "whatsapp_enabled": whatsapp_enabled,
                    "assets": assets if isinstance(assets, str) else _json.dumps(assets),
                    "min_probability": min_prob,
                    "daily_summary": daily,
                }
            return 1
        return orig_execute(query, params)

    monkeypatch.setattr(database, "fetch_one", fetch_one)
    monkeypatch.setattr(database, "execute", execute)
    auth_app.auth_limiter.events.clear()
    with TestClient(auth_app.app) as test_client:
        test_client.fake_db = fake
        yield test_client


def test_rules_threshold_and_channels():
    prefs = {"email_enabled": True, "whatsapp_enabled": False, "assets": ["RELIANCE"], "min_probability": 0.7}
    strong = {"asset": "RELIANCE", "prediction": 1, "probability_high_range": 0.8}
    weak = {"asset": "RELIANCE", "prediction": 1, "probability_high_range": 0.6}
    assert rules.should_notify(strong, prefs) == (True, "ok")
    assert rules.should_notify(weak, prefs)[0] is False
    assert rules.should_notify(strong, {"email_enabled": False, "whatsapp_enabled": False})[0] is False
    assert rules.should_notify({**strong, "asset": "TCS"}, prefs)[1] == "asset_not_subscribed"
    assert rules.should_notify({**strong, "prediction": 0}, prefs)[1] == "not_high_range"


def test_rules_session_gate():
    prefs = {"email_enabled": True, "assets": ["RELIANCE"], "min_probability": 0.6}
    pred = {"asset": "RELIANCE", "prediction": 1, "probability_high_range": 0.9}
    monday_open = datetime(2026, 8, 10, 4, 0, tzinfo=timezone.utc)  # 09:30 IST
    sunday = datetime(2026, 8, 9, 4, 0, tzinfo=timezone.utc)
    assert rules.should_notify(pred, prefs, monday_open)[0] is True
    assert rules.should_notify(pred, prefs, sunday) == (False, "outside_session")


def test_rules_whatsapp_validation():
    assert rules.is_valid_whatsapp("+919876543210") is True
    assert rules.is_valid_whatsapp("9876543210") is False
    with pytest.raises(ValueError):
        rules.validate_prefs({"whatsapp_enabled": True, "whatsapp_e164": None})


def test_register_with_reminder_optins(notif_client):
    response = notif_client.post(
        "/api/auth/register",
        json={
            "email": "rem@example.com",
            "password": "correct horse battery staple",
            "display_name": "Rem",
            "email_updates": False,
            "whatsapp_updates": True,
            "whatsapp_e164": "+919876543210",
        },
        headers=_csrf_headers(notif_client),
    )
    assert response.status_code == 201
    prefs = notif_client.get("/api/auth/me/notifications").json()
    assert prefs["emailEnabled"] is False
    assert prefs["whatsappEnabled"] is True
    assert prefs["whatsappNumber"] == "+919876543210"


def test_register_whatsapp_without_number_rejected(notif_client):
    response = notif_client.post(
        "/api/auth/register",
        json={
            "email": "rem2@example.com",
            "password": "correct horse battery staple",
            "display_name": "Rem",
            "whatsapp_updates": True,
        },
        headers=_csrf_headers(notif_client),
    )
    assert response.status_code == 422


def test_patch_notifications_roundtrip(notif_client):
    assert _register(notif_client).status_code == 201
    response = notif_client.patch(
        "/api/auth/me/notifications",
        json={"email_enabled": False, "assets": ["tcs"], "min_probability": 0.8, "daily_summary": True},
        headers=_csrf_headers(notif_client),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["emailEnabled"] is False
    assert body["assets"] == ["TCS"]
    assert body["minProbability"] == 0.8
    assert body["dailySummary"] is True


def test_patch_notifications_rejects_bad_asset_and_missing_number(notif_client):
    assert _register(notif_client).status_code == 201
    assert (
        notif_client.patch(
            "/api/auth/me/notifications",
            json={"assets": ["NOPE"]},
            headers=_csrf_headers(notif_client),
        ).status_code
        == 422
    )
    assert (
        notif_client.patch(
            "/api/auth/me/notifications",
            json={"whatsapp_enabled": True},
            headers=_csrf_headers(notif_client),
        ).status_code
        == 400
    )


def test_notifications_require_auth(notif_client):
    assert notif_client.get("/api/auth/me/notifications").status_code == 401
