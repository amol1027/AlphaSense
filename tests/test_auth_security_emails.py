"""Security email tests: welcome + sign-in alerts (no network, no MySQL)."""

from fastapi.testclient import TestClient

from src.auth import app as auth_app
from src.auth import database
from src.notifications import transactional
from tests.test_auth_api import FakeDB, _csrf_headers, _register


class MailboxFakeDB(FakeDB):
    def execute(self, query, params=()):
        flat = " ".join(query.split())
        if flat.startswith("UPDATE users SET whatsapp_e164"):
            return 0
        if flat.startswith("INSERT INTO notification_prefs"):
            return 1
        return super().execute(query, params)

    def fetch_one(self, query, params=()):
        flat = " ".join(query.split())
        if flat.startswith("SELECT email_enabled, whatsapp_enabled"):
            return None
        if flat.startswith("SELECT whatsapp_e164 FROM users"):
            return {"whatsapp_e164": None}
        return super().fetch_one(query, params)


def test_register_sends_welcome_email(monkeypatch):
    fake = MailboxFakeDB()
    sent = []
    monkeypatch.setattr(database, "fetch_one", fake.fetch_one)
    monkeypatch.setattr(database, "execute", fake.execute)
    monkeypatch.setattr(
        transactional, "send_security_email", lambda to, subject, body: sent.append((to, subject, body)) or True
    )
    auth_app.auth_limiter.events.clear()
    with TestClient(auth_app.app) as client:
        assert _register(client).status_code == 201
    assert len(sent) == 1
    to, subject, body = sent[0]
    assert to == "ada@example.com"
    assert "welcome" in subject.lower()
    assert "sign-in" in body.lower()


def test_login_sends_signin_alert(monkeypatch):
    fake = MailboxFakeDB()
    sent = []
    monkeypatch.setattr(database, "fetch_one", fake.fetch_one)
    monkeypatch.setattr(database, "execute", fake.execute)
    monkeypatch.setattr(
        transactional, "send_security_email", lambda to, subject, body: sent.append((to, subject, body)) or True
    )
    auth_app.auth_limiter.events.clear()
    with TestClient(auth_app.app) as client:
        assert _register(client).status_code == 201
        sent.clear()  # ignore the welcome email
        client.post("/api/auth/logout", headers=_csrf_headers(client))
        response = client.post(
            "/api/auth/login",
            json={"email": "ada@example.com", "password": "correct horse battery staple"},
            headers=_csrf_headers(client),
        )
        assert response.status_code == 200
    assert len(sent) == 1
    _, subject, body = sent[0]
    assert "sign-in" in subject.lower()
    assert "password" in body.lower()


def test_email_failure_never_breaks_auth(monkeypatch):
    fake = MailboxFakeDB()
    monkeypatch.setattr(database, "fetch_one", fake.fetch_one)
    monkeypatch.setattr(database, "execute", fake.execute)

    def boom(to, subject, body):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(transactional, "send_security_email", boom)
    auth_app.auth_limiter.events.clear()
    with TestClient(auth_app.app) as client:
        assert _register(client).status_code == 201
        client.post("/api/auth/logout", headers=_csrf_headers(client))
        assert (
            client.post(
                "/api/auth/login",
                json={"email": "ada@example.com", "password": "correct horse battery staple"},
                headers=_csrf_headers(client),
            ).status_code
            == 200
        )


def test_send_security_email_noop_without_smtp(monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert transactional.send_security_email("a@example.com", "s", "b") is False


def test_send_security_email_failure_returns_false(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_FROM", "a@example.com")

    class Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no network")

    monkeypatch.setattr("smtplib.SMTP", Broken)
    assert transactional.send_security_email("a@example.com", "s", "b") is False


def test_login_alert_copy_contains_time_and_ip():
    from datetime import datetime, timezone

    subject, body = transactional.format_login_alert_email(
        "Ada Lovelace", "1.2.3.4", datetime(2026, 8, 10, 4, 0, tzinfo=timezone.utc)
    )
    assert "09:30 IST" in body
    assert "1.2.3.4" in body
    assert "sign-in" in subject.lower()
