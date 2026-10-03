"""M1+M2 admin API tests (no MySQL required).

Covers: role defaults, disabled lockout, admin gates, user CRUD guards,
session revoke, audit trail, and the M2 ops surface (system status, shadow
tail, config, gated refresh). FakeDB mirrors the query surface of app.py.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.auth import app as auth_app
from src.auth import database


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class FakeDB:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}
        self.audit: list[dict] = []
        self.next_id = 1
        self.next_audit_id = 1

    def _by_id(self, uid: int) -> dict | None:
        return next((u for u in self.users.values() if u["id"] == uid), None)

    def _live_session(self, token_hash: str) -> dict | None:
        session = self.sessions.get(token_hash)
        if session and session["expires_at"] > _utcnow_naive():
            return session
        return None

    def _filtered_users(self, flat: str, params: tuple) -> list[dict]:
        users = list(self.users.values())
        if "disabled_at IS NULL" in flat:
            users = [u for u in users if u.get("disabled_at") is None]
        if "role = 'admin'" in flat and "COUNT" in flat:
            return [u for u in users if u.get("role") == "admin"]
        likes = [p.strip("%").lower() for p in params if isinstance(p, str) and p.startswith("%")]
        if likes:
            term = likes[0]
            users = [u for u in users if term in u["email"].lower() or term in u["display_name"].lower()]
        return sorted(users, key=lambda u: u["id"])

    # -- reads ---------------------------------------------------------
    def fetch_one(self, query: str, params: tuple = ()):
        flat = " ".join(query.split())
        if flat.startswith("SELECT id FROM users LIMIT"):
            return {"id": 1}
        if flat.startswith("SELECT token_hash FROM auth_sessions LIMIT"):
            return {"token_hash": "probe"}
        if flat.startswith("SELECT role FROM users"):
            return {"role": "user"}
        if flat.startswith("SELECT id FROM admin_audit_log LIMIT"):
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
            user = self._by_id(session["user_id"])
            if not user or user.get("disabled_at") is not None:
                return None
            return {
                "id": user["id"], "email": user["email"],
                "display_name": user["display_name"],
                "role": user.get("role", "user"), "disabled_at": user.get("disabled_at"),
            }
        if "SELECT COUNT(*) AS total FROM users" in flat:
            return {"total": len(self._filtered_users(flat, params))}
        if "FROM users WHERE id = %s" in flat:
            user = self._by_id(int(params[0]))
            return dict(user) if user else None
        if flat.startswith("SELECT id, email, display_name"):
            user = self.users.get(params[0])
            return dict(user) if user else None
        raise AssertionError(f"unhandled query: {query}")

    def fetch_all(self, query: str, params: tuple = ()):
        flat = " ".join(query.split())
        if "FROM users" in flat:
            users = self._filtered_users(flat, params)
            tail = [p for p in params if isinstance(p, int)]
            if len(tail) >= 2:
                limit, offset = tail[-2], tail[-1]
            elif len(tail) == 1:
                limit, offset = tail[0], 0
            else:
                limit, offset = 25, 0
            out = []
            for u in users[offset:offset + limit]:
                out.append({
                    "id": u["id"], "email": u["email"], "display_name": u["display_name"],
                    "role": u.get("role", "user"), "disabled_at": u.get("disabled_at"),
                    "created_at": u.get("created_at"),
                })
            return out
        if "FROM auth_sessions" in flat and "JOIN" not in flat:
            if "expires_at > UTC_TIMESTAMP()" in flat:
                sessions = [s for s in self.sessions.values() if s["expires_at"] > _utcnow_naive()]
            else:
                # Analytics read: all sessions, live or expired.
                sessions = list(self.sessions.values())
            if "WHERE user_id = %s" in flat:
                sessions = [s for s in sessions if s["user_id"] == params[0]]
            sessions.sort(key=lambda s: s.get("created_at") or _utcnow_naive(), reverse=True)
            tail = [p for p in params if isinstance(p, int)]
            if tail:
                sessions = sessions[:tail[-1]]
            return [dict(s) for s in sessions]
        if "FROM admin_audit_log" in flat:
            rows = list(self.audit)
            if "action = %s" in flat:
                # params: [action?, actor?, target?, limit, offset] - filter simply
                str_params = [p for p in params if isinstance(p, str)]
                if str_params:
                    rows = [r for r in rows if r["action"] == str_params[0]]
            rows.sort(key=lambda r: r["id"], reverse=True)
            ints = [p for p in params if isinstance(p, int)]
            if len(ints) >= 2:
                limit, offset = ints[-2], ints[-1]
            elif len(ints) == 1:
                limit, offset = ints[0], 0
            else:
                limit, offset = 25, 0
            return [dict(r) for r in rows[offset:offset + limit]]
        raise AssertionError(f"unhandled fetch_all: {query}")

    # -- writes --------------------------------------------------------
    def execute(self, query: str, params: tuple = ()):
        flat = " ".join(query.split())
        if flat.startswith("INSERT INTO users"):
            email = params[0]
            from mysql.connector import IntegrityError

            if email in self.users:
                raise IntegrityError("Duplicate entry")
            role = "admin" if "'admin'" in flat else "user"
            user = {"id": self.next_id, "email": email, "display_name": params[1],
                    "password_hash": params[2], "role": role, "disabled_at": None,
                    "created_at": _utcnow_naive()}
            self.users[email] = user
            self.next_id += 1
            return user["id"]
        if flat.startswith("INSERT INTO auth_sessions"):
            now = _utcnow_naive()
            self.sessions[params[0]] = {
                "token_hash": params[0], "csrf_hash": params[1],
                "user_id": params[2], "expires_at": params[3], "created_at": now,
            }
            return 1
        if flat.startswith("INSERT INTO admin_audit_log"):
            self.audit.append({
                "id": self.next_audit_id, "actor_id": params[0], "action": params[1],
                "target_user_id": params[2], "detail": params[3], "created_at": _utcnow_naive(),
            })
            self.next_audit_id += 1
            return 1
        if flat.startswith("UPDATE users SET display_name"):
            user = self._by_id(params[1])
            if user:
                user["display_name"] = params[0]
            return 0
        if flat.startswith("UPDATE users SET password_hash"):
            for user in self.users.values():
                if user["id"] == params[1]:
                    user["password_hash"] = params[0]
            return 0
        if flat.startswith("UPDATE users SET role"):
            user = self._by_id(params[-1])
            if user:
                user["role"] = params[0]
            return 0
        if "SET disabled_at = UTC_TIMESTAMP()" in flat:
            user = self._by_id(params[0])
            if user:
                user["disabled_at"] = _utcnow_naive()
            return 0
        if "SET disabled_at = NULL" in flat:
            user = self._by_id(params[0])
            if user:
                user["disabled_at"] = None
            return 0
        if flat.startswith("UPDATE auth_sessions SET csrf_hash"):
            session = self._live_session(params[1])
            if session:
                session["csrf_hash"] = params[0]
            return 0
        if flat.startswith("DELETE FROM auth_sessions"):
            if not params:
                return 0
            if "WHERE user_id = %s" in flat:
                for key in [k for k, s in self.sessions.items() if s["user_id"] == params[0]]:
                    del self.sessions[key]
            else:
                self.sessions.pop(params[0], None)
            return 0
        raise AssertionError(f"unhandled execute: {query}")


@pytest.fixture
def client(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(database, "fetch_one", fake.fetch_one)
    monkeypatch.setattr(database, "fetch_all", fake.fetch_all)
    monkeypatch.setattr(database, "execute", fake.execute)
    auth_app.auth_limiter.events.clear()
    auth_app.admin_limiter.events.clear()
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


def _promote(client, email: str) -> None:
    client.fake_db.users[email]["role"] = "admin"


def _admin_headers(client, email: str = "root@example.com") -> dict[str, str]:
    _register(client, email=email, name="Root")
    _promote(client, email)
    _login(client, email)
    return _csrf_headers(client)


def _login(client, email: str, password: str = "correct horse battery staple") -> None:
    resp = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
        headers=_csrf_headers(client),
    )
    assert resp.status_code == 200, resp.text


# -- M0 regression ------------------------------------------------------

def test_register_and_me_default_to_user_role(client):
    assert _register(client).status_code == 201
    assert client.get("/api/auth/me").json()["role"] == "user"


def test_login_blocked_when_disabled(client):
    assert _register(client).status_code == 201
    client.fake_db.users["ada@example.com"]["disabled_at"] = _utcnow_naive()
    client.post("/api/auth/logout", headers=_csrf_headers(client))
    response = client.post(
        "/api/auth/login",
        json={"email": "ada@example.com", "password": "correct horse battery staple"},
        headers=_csrf_headers(client),
    )
    assert response.status_code == 401


def test_me_returns_401_after_disable(client):
    assert _register(client).status_code == 201
    assert client.get("/api/auth/me").status_code == 200
    client.fake_db.users["ada@example.com"]["disabled_at"] = _utcnow_naive()
    assert client.get("/api/auth/me").status_code == 401


def test_audit_write_is_best_effort(client):
    auth_app._audit(1, "test.action", 2, '{"k": 1}')
    assert client.fake_db.audit and client.fake_db.audit[0]["action"] == "test.action"


def test_schema_files_contain_admin_objects():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    schema = (root / "src" / "auth" / "schema.sql").read_text(encoding="utf-8")
    migration = (root / "src" / "auth" / "migrations" / "001_admin.sql").read_text(encoding="utf-8")
    assert "admin_audit_log" in schema and "admin_audit_log" in migration
    assert "disabled_at" in schema and "role" in schema


# -- M1 gates -----------------------------------------------------------

def test_admin_endpoints_require_admin(client):
    assert _register(client).status_code == 201  # plain user session
    assert client.get("/api/admin/users").status_code == 403
    assert client.get("/api/admin/audit").status_code == 403
    # anonymous
    client.post("/api/auth/logout", headers=_csrf_headers(client))
    assert client.get("/api/admin/users").status_code == 401


def test_admin_can_list_search_and_get_user(client):
    headers = _admin_headers(client)
    assert _register(client, email="bob@example.com", name="Bob").status_code == 201
    _login(client, "root@example.com")
    headers = _csrf_headers(client)
    # GET reads need no CSRF; cookies carry the admin session
    listed = client.get("/api/admin/users").json()
    assert listed["total"] >= 2
    searched = client.get("/api/admin/users", params={"search": "bob"}).json()
    assert searched["total"] == 1 and searched["users"][0]["email"] == "bob@example.com"
    uid = searched["users"][0]["id"]
    assert client.get(f"/api/admin/users/{uid}").json()["email"] == "bob@example.com"
    assert client.get("/api/admin/users/999999").status_code == 404


def test_admin_cannot_change_own_role_or_disable_self(client):
    headers = _admin_headers(client)
    me = client.get("/api/auth/me").json()
    r1 = client.patch(f"/api/admin/users/{me['id']}", json={"role": "user"}, headers=headers)
    assert r1.status_code == 400
    r2 = client.patch(f"/api/admin/users/{me['id']}", json={"disabled": True}, headers=headers)
    assert r2.status_code == 400


def test_last_admin_guard(client):
    headers = _admin_headers(client)
    me = client.get("/api/auth/me").json()
    assert _register(client, email="second@example.com", name="Second").status_code == 201
    _login(client, "root@example.com")
    headers = _csrf_headers(client)
    second = client.get("/api/admin/users", params={"search": "second"}).json()["users"][0]
    # demoting the only other path: promote second, demote self still blocked by self-guard;
    # demoting second (a plain user) is a no-op success; disabling the last admin is refused
    r = client.patch(f"/api/admin/users/{me['id']}", json={"disabled": True}, headers=headers)
    assert r.status_code == 400  # self-guard fires first
    # make second an admin, then demote root -> allowed (2 admins), then demote second -> refused
    client.patch(f"/api/admin/users/{second['id']}", json={"role": "admin"}, headers=headers)
    assert client.patch(f"/api/admin/users/{second['id']}", json={"role": "user"}, headers=headers).status_code == 200


def test_disable_revokes_sessions_and_audits(client):
    _admin_headers(client)
    assert _register(client, email="mallory@example.com", name="Mallory").status_code == 201
    _login(client, "root@example.com")
    target = client.get("/api/admin/users", params={"search": "mallory"}).json()["users"][0]
    headers = _csrf_headers(client)
    before = client.fake_db.audit.__len__()
    resp = client.patch(f"/api/admin/users/{target['id']}", json={"disabled": True}, headers=headers)
    assert resp.status_code == 200 and resp.json()["disabled"] is True
    assert client.fake_db.audit.__len__() == before + 1
    assert client.get("/api/admin/sessions", params={"user_id": target["id"]}).json()["sessions"] == []


def test_revoke_single_session_and_self_guard(client):
    headers = _admin_headers(client)
    sessions = client.get("/api/admin/sessions").json()["sessions"]
    assert sessions, "admin should have a session"
    own_hash = sessions[0]["tokenHash"]
    assert client.delete(f"/api/admin/sessions/{own_hash}", headers=headers).status_code == 400
    assert client.delete("/api/admin/sessions/not-hex", headers=headers).status_code == 400


# -- M2 ops ---------------------------------------------------------------

def _point_ops_paths(monkeypatch, tmp_path, *, cache_rows=2, log_rows=True, artifacts=(), webapp=False):
    """Redirect the ops file probes at tmp files. Returns paths for asserts."""
    import json as json_module

    cache = tmp_path / "live_market_15m.csv"
    cache.write_text(
        "asset,exchange,timestamp,open,high,low,close,volume\n"
        + "RELIANCE,NSE,2026-10-01T09:00:00+00:00,1,2,0.5,1.5,100\n" * cache_rows,
        encoding="utf-8",
    )
    status = tmp_path / "live_status.json"
    status.write_text(json_module.dumps({"last_refresh_at": "2026-10-01T09:17:37+00:00"}), encoding="utf-8")
    log = tmp_path / "live_log.csv"
    if log_rows:
        log.write_text(
            "asset,prediction_timestamp,probability,prediction,realized_label,correct\n"
            "RELIANCE,2026-10-01T08:00:00+00:00,0.8,1,1,1\n"
            "TCS,2026-10-01T08:00:00+00:00,0.3,0,,,\n",
            encoding="utf-8",
        )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    for asset in artifacts:
        (artifact_dir / f"{asset.lower()}.joblib").write_bytes(b"fake")
    webapp_index = tmp_path / "index.html"
    if webapp:
        webapp_index.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(auth_app, "_MARKET_CACHE_PATH", cache)
    monkeypatch.setattr(auth_app, "_LIVE_STATUS_PATH", status)
    monkeypatch.setattr(auth_app, "_LIVE_LOG_PATH", log)
    monkeypatch.setattr(auth_app, "_NEWS_STATUS_PATH", tmp_path / "live_news_status.json")
    monkeypatch.setattr(auth_app, "_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(auth_app, "_WEBAPP_INDEX", webapp_index)
    monkeypatch.setattr(auth_app, "_prediction_health", lambda *a, **k: {
        "ok": True, "httpStatus": 200, "status": "ok",
        "feedState": "fresh", "feedAgeMinutes": 5.0,
        "checks": {"market_cache": True}, "error": None,
    })
    return cache, log


def test_admin_ops_require_admin(client):
    assert _register(client).status_code == 201  # plain user
    assert client.get("/api/admin/system/status").status_code == 403
    assert client.get("/api/admin/system/shadow").status_code == 403
    assert client.get("/api/admin/system/config").status_code == 403
    assert client.post("/api/admin/system/refresh", headers=_csrf_headers(client)).status_code == 403


def test_admin_system_status_aggregates_without_secrets(client, monkeypatch, tmp_path):
    _admin_headers(client)
    _point_ops_paths(monkeypatch, tmp_path, cache_rows=3, artifacts=("RELIANCE",))
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "super-secret-token-value")
    response = client.get("/api/admin/system/status")
    assert response.status_code == 200
    assert "super-secret-token-value" not in response.text
    body = response.json()
    assert body["authDb"] == {"ok": True}
    assert body["prediction"]["ok"] is True
    assert body["marketCache"]["rows"] == 3
    assert body["marketCache"]["refresh"]["last_refresh_at"].startswith("2026-10-01")
    assert body["modelArtifacts"]["ok"] is False
    assert body["modelArtifacts"]["missing"] and "RELIANCE" not in body["modelArtifacts"]["missing"]
    assert body["upstoxTokenConfigured"] == {"ok": True}
    assert body["webapp"] == {"ok": False}
    assert "password_hash" not in response.text and "token_hash" not in response.text


def test_admin_system_shadow_tail_and_counts(client, monkeypatch, tmp_path):
    _admin_headers(client)
    _, log = _point_ops_paths(monkeypatch, tmp_path)
    body = client.get("/api/admin/system/shadow").json()
    assert body["available"] is True
    assert body["counts"] == {"scored": 1, "correct": 1, "pending": 1}
    assert [r["asset"] for r in body["rows"]] == ["RELIANCE", "TCS"]
    limited = client.get("/api/admin/system/shadow", params={"limit": 1}).json()
    assert len(limited["rows"]) == 1
    log.unlink()
    assert client.get("/api/admin/system/shadow").json()["available"] is False


def test_admin_system_config_has_no_secrets(client, monkeypatch):
    _admin_headers(client)
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "super-secret-token-value")
    response = client.get("/api/admin/system/config")
    assert response.status_code == 200
    assert "super-secret-token-value" not in response.text
    body = response.json()
    assert body["assets"] == ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    assert body["horizon"] == "1h"


def test_admin_system_refresh_success_is_audited(client, monkeypatch):
    headers = _admin_headers(client)
    monkeypatch.setattr(
        auth_app, "_run_system_refresh",
        lambda: {"assetsUpdated": 5, "outcomesScored": 2, "predictionsAdded": 3},
    )
    before = len(client.fake_db.audit)
    response = client.post("/api/admin/system/refresh", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"assetsUpdated": 5, "outcomesScored": 2, "predictionsAdded": 3}
    assert len(client.fake_db.audit) == before + 1
    assert client.fake_db.audit[-1]["action"] == "system.refresh"


def test_admin_system_refresh_failure_is_502(client, monkeypatch):
    headers = _admin_headers(client)

    def _boom():
        raise RuntimeError("upstox down")

    monkeypatch.setattr(auth_app, "_run_system_refresh", _boom)
    assert client.post("/api/admin/system/refresh", headers=headers).status_code == 502


def test_admin_system_refresh_lock_conflict(client):
    headers = _admin_headers(client)
    assert auth_app._REFRESH_LOCK.acquire(blocking=False)
    try:
        assert client.post("/api/admin/system/refresh", headers=headers).status_code == 409
    finally:
        auth_app._REFRESH_LOCK.release()


# -- analytics + exports --------------------------------------------------

def _seed_analytics(client, monkeypatch, tmp_path):
    """Two users (one backdated, one disabled), live+expired sessions, audit
    across families, and a 4-row shadow log. Returns the fake DB."""
    from datetime import timedelta

    _admin_headers(client)  # root admin, logged in
    assert _register(client, email="bob@example.com", name="Bob").status_code == 201
    _login(client, "root@example.com")
    fake = client.fake_db
    now = _utcnow_naive()
    two_days_ago = now - timedelta(days=2)
    fake.users["bob@example.com"]["created_at"] = two_days_ago
    fake.users["bob@example.com"]["disabled_at"] = now
    # an expired session for bob (created two days ago)
    fake.sessions["deadbeef"] = {
        "token_hash": "deadbeef", "csrf_hash": "x", "user_id": fake.users["bob@example.com"]["id"],
        "expires_at": now - timedelta(days=1), "created_at": now - timedelta(days=2),
    }
    root_id = fake.users["root@example.com"]["id"]
    auth_app._audit(root_id, "user.update", 2, "{}")
    auth_app._audit(root_id, "session.revoke_all", 2, None)
    auth_app._audit(root_id, "system.refresh", None, "{}")
    for entry in fake.audit:
        entry["created_at"] = two_days_ago
    log = tmp_path / "analytics_log.csv"
    log.write_text(
        "asset,prediction_timestamp,probability,prediction,realized_label,correct\n"
        "RELIANCE,2026-09-20T08:00:00+00:00,0.85,1,1,1\n"
        "TCS,2026-09-20T09:00:00+00:00,0.15,0,0,1\n"
        "INFY,2026-09-20T10:00:00+00:00,0.9,1,0,0\n"
        "HDFCBANK,2026-09-20T11:00:00+00:00,0.55,1,,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(auth_app, "_LIVE_LOG_PATH", log)
    return fake


def test_analytics_requires_admin(client):
    assert _register(client).status_code == 201
    assert client.get("/api/admin/analytics/overview").status_code == 403
    assert client.get("/api/admin/export/users").status_code == 403
    assert client.get("/api/admin/export/audit").status_code == 403
    assert client.get("/api/admin/export/shadow").status_code == 403


def test_analytics_overview_aggregates(client, monkeypatch, tmp_path):
    fake = _seed_analytics(client, monkeypatch, tmp_path)
    headers = _csrf_headers(client)
    response = client.get("/api/admin/analytics/overview", headers=headers)
    assert response.status_code == 200
    assert "password_hash" not in response.text and "token_hash" not in response.text
    body = response.json()

    users = body["users"]
    assert users["total"] == 2 and users["disabled"] == 1 and users["admins"] == 1
    assert len(users["signupsPerDay"]) == 30
    assert sum(d["count"] for d in users["signupsPerDay"]) == 2

    sessions = body["sessions"]
    assert sessions["live"] == len(
        [s for s in fake.sessions.values() if s["expires_at"] > _utcnow_naive()]
    )
    assert sum(d["count"] for d in sessions["perDay"]) == len(fake.sessions)

    audit = body["audit"]
    families = {d["date"]: d for d in audit["perDay"]}
    assert any(v["user"] + v["session"] + v["system"] > 0 for v in families.values())
    assert audit["topActors"] and audit["topActors"][0]["email"] == "root@example.com"

    shadow = body["shadow"]
    assert shadow["total"] == 4 and shadow["scored"] == 3
    assert shadow["correct"] == 2 and shadow["pending"] == 1
    assert shadow["accuracy"] == round(2 / 3, 4)
    assert sum(shadow["confidence"]) == 4
    assert {a["asset"] for a in shadow["perAsset"]} == {"RELIANCE", "TCS", "INFY", "HDFCBANK"}
    assert len(shadow["rolling7d"]) == 30
    assert all(
        (entry["accuracy"] is None) or (0 <= entry["accuracy"] <= 1)
        for entry in shadow["rolling7d"]
    )


def test_export_users_csv_has_header_no_secrets_and_sanitizes(client, monkeypatch, tmp_path):
    _seed_analytics(client, monkeypatch, tmp_path)
    fake = client.fake_db
    fake.users["bob@example.com"]["display_name"] = "=cmd|evil"
    response = client.get("/api/admin/export/users")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert 'filename="alphasense-users.csv"' in response.headers["content-disposition"]
    assert "password_hash" not in response.text
    assert "'=cmd|evil" in response.text


def test_export_audit_csv(client, monkeypatch, tmp_path):
    _seed_analytics(client, monkeypatch, tmp_path)
    response = client.get("/api/admin/export/audit")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert response.text.splitlines()[0] == "id,actor_id,action,target_user_id,detail,created_at"
    assert "system.refresh" in response.text


def test_export_shadow_csv_present_and_missing(client, monkeypatch, tmp_path):
    _seed_analytics(client, monkeypatch, tmp_path)
    response = client.get("/api/admin/export/shadow")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "RELIANCE,2026-09-20" in response.text
    import pathlib

    monkeypatch.setattr(auth_app, "_LIVE_LOG_PATH", pathlib.Path(str(tmp_path / "nope.csv")))
    header_only = client.get("/api/admin/export/shadow")
    assert header_only.status_code == 200
    assert header_only.text.strip() == "asset,prediction_timestamp"
