"""Account API contract tests (no MySQL required).

The database module is swapped for an in-memory fake covering exactly the
query surface used by src/auth/app.py. The per-process rate limiter is reset
between tests because TestClient always presents the same client IP.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from mysql.connector import IntegrityError

from src.auth import app as auth_app
from src.auth import database


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class FakeDB:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}
        self.next_id = 1

    def _live_session(self, token_hash: str) -> dict | None:
        session = self.sessions.get(token_hash)
        if session and session["expires_at"] > _utcnow_naive():
            return session
        return None

    def fetch_one(self, query: str, params: tuple = ()):
        flat = " ".join(query.split())
        if flat.startswith("SELECT id FROM users"):
            return {"id": 1}
        if flat.startswith("SELECT token_hash FROM auth_sessions"):
            return {"token_hash": "probe"}
        if flat.startswith("SELECT role FROM users"):
            return {"role": "user"}
        if flat.startswith("SELECT id FROM admin_audit_log"):
            return {"id": 1}
        if flat == "SELECT 1":
            return {"1": 1}
        if flat.startswith("SELECT csrf_hash FROM auth_sessions"):
            session = self._live_session(params[0])
            return {"csrf_hash": session["csrf_hash"]} if session else None
        if "FROM auth_sessions s JOIN users" in flat:
            session = self._live_session(params[0])
            if not session:
                return None
            user = next((u for u in self.users.values() if u["id"] == session["user_id"]), None)
            if not user or user.get("disabled_at") is not None:
                return None
            return {"id": user["id"], "email": user["email"], "display_name": user["display_name"],
                    "role": user.get("role", "user"), "disabled_at": user.get("disabled_at")}
        if flat.startswith("SELECT id, email, display_name, password_hash"):
            key = params[0]
            user = self.users.get(key)
            if user is None:
                user = next((u for u in self.users.values() if u["id"] == key), None)
            return dict(user) if user else None
        raise AssertionError(f"unhandled query: {query}")

    def execute(self, query: str, params: tuple = ()):
        flat = " ".join(query.split())
        if flat.startswith("INSERT INTO users"):
            email = params[0]
            if email in self.users:
                raise IntegrityError("Duplicate entry")
            user = {"id": self.next_id, "email": email, "display_name": params[1], "password_hash": params[2]}
            self.users[email] = user
            self.next_id += 1
            return user["id"]
        if flat.startswith("INSERT INTO auth_sessions"):
            self.sessions[params[0]] = {
                "token_hash": params[0], "csrf_hash": params[1],
                "user_id": params[2], "expires_at": params[3],
            }
            return 1
        if flat.startswith("INSERT INTO admin_audit_log"):
            return 1
        if flat.startswith("UPDATE users SET password_hash"):
            for user in self.users.values():
                if user["id"] == params[1]:
                    user["password_hash"] = params[0]
            return 0
        if flat.startswith("UPDATE users SET display_name"):
            for user in self.users.values():
                if user["id"] == params[1]:
                    user["display_name"] = params[0]
            return 0
        if flat.startswith("UPDATE auth_sessions SET csrf_hash"):
            session = self._live_session(params[1])
            if session:
                session["csrf_hash"] = params[0]
            return 0
        if flat.startswith("DELETE FROM auth_sessions"):
            if params:
                self.sessions.pop(params[0], None)
            return 0
        raise AssertionError(f"unhandled query: {query}")


@pytest.fixture
def client(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(database, "fetch_one", fake.fetch_one)
    monkeypatch.setattr(database, "execute", fake.execute)
    auth_app.auth_limiter.events.clear()
    with TestClient(auth_app.app) as test_client:
        test_client.fake_db = fake
        yield test_client


def _csrf_headers(client) -> dict[str, str]:
    token = client.get("/api/auth/csrf").json()["csrfToken"]
    return {"Origin": "http://127.0.0.1:5173", "X-CSRF-Token": token}


def _register(client, email="ada@example.com", password="correct horse battery staple", name="Ada"):
    return client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "display_name": name},
        headers=_csrf_headers(client),
    )


def test_register_login_me_logout_roundtrip(client):
    assert _register(client).status_code == 201
    assert client.get("/api/auth/me").status_code == 200
    assert client.get("/api/auth/me").json()["email"] == "ada@example.com"
    assert client.post("/api/auth/logout", headers=_csrf_headers(client)).status_code == 200
    assert client.get("/api/auth/me").status_code == 401


def test_register_duplicate_email_returns_409_without_leaking(client):
    assert _register(client).status_code == 201
    response = _register(client)
    assert response.status_code == 409
    assert "already" not in response.json()["detail"].lower()


def test_login_wrong_password_returns_401(client):
    assert _register(client).status_code == 201
    client.post("/api/auth/logout", headers=_csrf_headers(client))
    response = client.post(
        "/api/auth/login",
        json={"email": "ada@example.com", "password": "wrong password here"},
        headers=_csrf_headers(client),
    )
    assert response.status_code == 401


def test_login_unknown_email_returns_same_401(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "wrong password here"},
        headers=_csrf_headers(client),
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Email or password is incorrect"


def test_me_without_session_returns_401(client):
    assert client.get("/api/auth/me").status_code == 401


def test_register_rejects_short_and_common_passwords(client):
    assert _register(client, password="short1234").status_code == 422
    assert _register(client, password="password123").status_code == 422
    assert _register(client, email="bad-email", password="longenoughpassword").status_code == 422


def test_login_accepts_legacy_short_password(client):
    # Accounts created under the old 6-char policy must keep working.
    hashed = auth_app.PASSWORD_HASHER.hash("old6pw")
    client.fake_db.users["vet@example.com"] = {
        "id": 99, "email": "vet@example.com",
        "display_name": "Vet", "password_hash": hashed,
    }
    response = client.post(
        "/api/auth/login",
        json={"email": "vet@example.com", "password": "old6pw"},
        headers=_csrf_headers(client),
    )
    assert response.status_code == 200


def test_missing_csrf_token_returns_403(client):
    client.get("/api/auth/csrf")
    response = client.post(
        "/api/auth/register",
        json={"email": "x@example.com", "password": "longenoughpassword", "display_name": "X"},
        headers={"Origin": "http://127.0.0.1:5173"},
    )
    assert response.status_code == 403


def test_disallowed_origin_returns_403(client):
    token = client.get("/api/auth/csrf").json()["csrfToken"]
    response = client.post(
        "/api/auth/login",
        json={"email": "a@example.com", "password": "whatever password"},
        headers={"Origin": "https://evil.example", "X-CSRF-Token": token},
    )
    assert response.status_code == 403


def test_rate_limiter_returns_429_with_retry_after(client):
    assert _register(client).status_code == 201
    client.post("/api/auth/logout", headers=_csrf_headers(client))
    headers = _csrf_headers(client)
    statuses = {
        client.post(
            "/api/auth/login",
            json={"email": "ada@example.com", "password": "wrong password here"},
            headers=headers,
        ).status_code
        for _ in range(12)
    }
    assert 429 in statuses


def test_health_returns_200(client):
    assert client.get("/api/auth/health").status_code == 200


def test_patch_me_updates_display_name(client):
    assert _register(client).status_code == 201
    response = client.patch(
        "/api/auth/me",
        json={"display_name": "  Ada   Lovelace  "},
        headers=_csrf_headers(client),
    )
    assert response.status_code == 200
    assert response.json()["displayName"] == "Ada Lovelace"
    assert client.get("/api/auth/me").json()["displayName"] == "Ada Lovelace"


def test_patch_me_rejects_blank_name_and_anonymous(client):
    assert _register(client).status_code == 201
    assert client.patch(
        "/api/auth/me",
        json={"display_name": "   "},
        headers=_csrf_headers(client),
    ).status_code == 422
    client.post("/api/auth/logout", headers=_csrf_headers(client))
    assert client.patch(
        "/api/auth/me",
        json={"display_name": "Ada"},
        headers=_csrf_headers(client),
    ).status_code == 401


def test_password_change_roundtrip(client):
    assert _register(client).status_code == 201
    headers = _csrf_headers(client)
    assert client.post(
        "/api/auth/me/password",
        json={"current_password": "correct horse battery staple", "new_password": "a brand new passphrase 99"},
        headers=headers,
    ).status_code == 200
    client.post("/api/auth/logout", headers=_csrf_headers(client))
    assert client.post(
        "/api/auth/login",
        json={"email": "ada@example.com", "password": "correct horse battery staple"},
        headers=_csrf_headers(client),
    ).status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"email": "ada@example.com", "password": "a brand new passphrase 99"},
        headers=_csrf_headers(client),
    ).status_code == 200


def test_password_change_rejects_wrong_current_and_weak_new(client):
    assert _register(client).status_code == 201
    assert client.post(
        "/api/auth/me/password",
        json={"current_password": "wrong password here", "new_password": "a brand new passphrase 99"},
        headers=_csrf_headers(client),
    ).status_code == 401
    assert client.post(
        "/api/auth/me/password",
        json={"current_password": "correct horse battery staple", "new_password": "password123"},
        headers=_csrf_headers(client),
    ).status_code == 422
    assert client.post(
        "/api/auth/me/password",
        json={"current_password": "correct horse battery staple", "new_password": "correct horse battery staple"},
        headers=_csrf_headers(client),
    ).status_code == 400
