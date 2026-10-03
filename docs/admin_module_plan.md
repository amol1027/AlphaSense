# Admin Module — Plan

## 1. Goal

Add a minimal, secure admin surface for operating AlphaSense without changing research guarantees:

- Manage accounts (`users`, `auth_sessions` in MySQL).
- Observe/operate the three local services (prediction API `:8000`, auth API `:8010`, shadow runner) and the frozen volatility pipeline.
- Audit every privileged action.

Non-goals (must not break):

- No change to locked holdout (`2026-08-10` onward, scored-once), frozen 9-feature 1h volatility model, temporal/session contracts, or Phase 1–4 frozen results.
- No live trading, no investment advice, no new signal research in this module.
- No change to public research pages becoming private; `/markets` stays auth-gated as today (`website/src/main.tsx`).

## 2. Current state (verified in repo)

| Area | What exists | Gap for admin |
|---|---|---|
| Auth API `src/auth/app.py` | register/login/me/logout, cookie `alpha_session` + double-submit CSRF `alpha_csrf`, Argon2id, per-process sliding-window limiter (single worker required), `GET /api/auth/health`, `/csrf` | No roles, no user listing, no disable/revoke, no audit log |
| Auth DB `src/auth/schema.sql` + `database.py` | `users(id,email,display_name,password_hash,created_at)`, `auth_sessions(token_hash,csrf_hash,user_id,expires_at,...)`; pool size 5 | No `role` / `disabled` columns, no `admin_audit_log` table |
| Prediction API `src/prediction/server.py` | stdlib `ThreadingHTTPServer`: `GET /health` (per-check 200/503), `/dashboard`, `/chart`, `/stream`, `/predict`, `POST /refresh` (origin-checked, single-flight lock) | No auth at all; `/refresh` is open to allowed origins — admin must not expose it publicly without a gate |
| Website `website/src/` | Routes `/`, `/markets` (+`/dashboard` alias), `/login`, `/register` in `main.tsx`; `auth-client.ts` CSRF flow; `MarketDashboard.tsx` calls `/dashboard`, `/chart`, `/stream`, `/refresh` | No `/admin` route, no role-aware nav, `safeNext()` allowlist is `/`, `/markets`, `/dashboard` only |
| Ops | `scripts/run_shadow.py`, `refresh_live_market.py`, `log_live_prediction.py`, `check_upstox_token.py`, `initialize_auth_db.py`; rolling cache `data/interim/live_*.csv/json`; `deploy/systemd/` units; docs `STARTING_THE_APP.md`, `ACCOUNT_AUTH.md` | No admin-triggered ops with authz/audit; token validity only gated at runner boot |
| Tests | `tests/test_auth_api.py` uses in-memory `FakeDB` over exact query surface; suite 256 passed | New queries/endpoints need FakeDB + migration tests |

## 3. Roles and access model

- Two roles only (keep it small): `user` (default), `admin`.
- Storage: `users.role ENUM('user','admin') DEFAULT 'user'`, `users.disabled_at DATETIME NULL` (soft-disable preserves history; hard delete is out of scope initially).
- Enforcement: server-side on every admin endpoint via `_current_user()` + role/disabled check. Never trust a frontend flag.
- Disabled users: `login` rejected, existing sessions invalidated (delete from `auth_sessions`), `/me` returns 401.
- Bootstrap: first admin created by CLI script (see §7), not by public registration. Registration always creates `role='user'`.
- Frontend: `/admin` route renders only when `GET /api/auth/me` returns `role:'admin'`; otherwise redirect to `/login` (same gating pattern as `/markets`).

## 4. Scope (phased)

### M0 — Schema + roles + bootstrap (backend only, no UI)
1. Migration `src/auth/migrations/001_admin.sql`: add `role`, `disabled_at` to `users`; create `admin_audit_log(id, actor_id, action, target_user_id NULL, detail JSON NULL, created_at)`.
2. `scripts/create_admin.py`: interactive (getpass) create-or-promote by email; refuses to demote/disable the last active admin.
3. Update `check_schema()` to probe new columns/tables; update `initialize_auth_db.py` path.
4. Tests: migration idempotence, login blocked when disabled, last-admin guard.

### M1 — User & session administration (core admin API + UI)
API (all under auth service `:8010`, all require admin session + valid CSRF + same-origin, all write to `admin_audit_log`):

```text
GET  /api/admin/users?search=&limit=&offset=&include_disabled=
GET  /api/admin/users/{id}
PATCH /api/admin/users/{id}            {display_name?, role?, disabled?: bool}
POST /api/admin/users/{id}/revoke-sessions
GET  /api/admin/sessions?user_id=&limit=
DELETE /api/admin/sessions/{token_hash_prefix...}  (or by user_id= for bulk revoke)
GET  /api/admin/audit?action=&actor=&limit=&offset=
GET  /api/auth/me  → add {role, disabled}
```

Rules:
- `PATCH role`: only `user↔admin`; cannot change own role or disable self.
- `PATCH disabled=true`: deletes that user's rows in `auth_sessions` in same transaction.
- List never returns `password_hash`; search by email prefix + display_name LIKE, paginated (default 25, max 100).
- Rate limit: reuse `auth_limiter` bucket with a separate stricter key (e.g. 30/min/admin) + keep login/register limiter untouched.

UI (`website/src/AdminDashboard.tsx`, lazy-loaded like `MarketDashboard`):
- `/admin` route + header link visible only to admins; add `/admin` to `safeNext()` allowlist.
- Tabs: Users (table + search + disable/enable + promote/demote + revoke sessions), Sessions (by user), Audit log (read-only table).
- All mutations send `X-CSRF-Token` via `auth-client.ts` `getCsrfToken()`; reuse `readError()` patterns.

### M2 — System operations (read-mostly, tightly gated actions)
Read endpoints (admin-only) that aggregate what operators today check manually:

```text
GET /api/admin/system/status
  → proxies prediction GET /health, auth DB probe, shadow freshness
    (live_status.json + live_market_15m.csv mtime/row count), model artifact presence
    (src/prediction/artifacts.py artifact_path per KNOWN_ASSETS), upstox token *presence*
    (never the value), systemd unit state if available
GET /api/admin/system/shadow?limit=50   → tail of live_log.csv + scoring counts
GET /api/admin/system/config            → non-secret config only
    (assets, horizons, allowed origins hostnames, artifact paths, calendar files)
POST /api/admin/system/refresh          → server-side call of the existing
    refresh_market_data()+label+log flow (same code as prediction POST /refresh),
    single-flight lock reused, audited; does NOT accept bars from the client
```

Constraints:
- No artifact upload/retrain from admin UI in this plan (frozen model). Show `trained_at`, `n_train`, missing-asset list; retraining stays a CLI + code-review action (`scripts/train_volatility_model.py`).
- No secret display: `UPSTOX_ACCESS_TOKEN`, `MYSQL_PASSWORD` never returned; show boolean `configured` only.
- Prediction server itself stays unauthenticated locally; admin UI calls ops through the auth service, which allowlists `127.0.0.1:8000` server-side. Do not add CORS/auth to the stdlib prediction server in M2.

### M3 — Hardening & audit
- Every `PATCH/POST/DELETE` in §M1–M2 writes `admin_audit_log` (actor, action, target, detail) in the same DB transaction; audit table is append-only for app user (no UPDATE/DELETE grants).
- Add `GET /api/admin/audit` pagination + tests proving a disabled admin's session stops working mid-flight.
- Docs: extend `docs/ACCOUNT_AUTH.md` with admin bootstrap + production notes (single worker still required; edge rate limit; HTTPS + `AUTH_COOKIE_SECURE=true`; MySQL least-privilege GRANT update for new tables).
- `deploy/systemd/` : no new service; document admin as part of auth API unit.

Explicitly deferred: password reset, email verification, account self-deletion, bulk CSV import, holiday-calendar editor (read-only path display only), per-asset model toggles.

## 5. DB design

```sql
ALTER TABLE users
  ADD COLUMN role ENUM('user','admin') NOT NULL DEFAULT 'user',
  ADD COLUMN disabled_at DATETIME NULL,
  ADD KEY ix_users_role_disabled (role, disabled_at);

CREATE TABLE IF NOT EXISTS admin_audit_log (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  actor_id BIGINT UNSIGNED NOT NULL,
  action VARCHAR(64) NOT NULL,
  target_user_id BIGINT UNSIGNED NULL,
  detail JSON NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY ix_audit_actor (actor_id),
  KEY ix_audit_created (created_at),
  CONSTRAINT fk_audit_actor FOREIGN KEY (actor_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
```

GRANT update for app user: add `SELECT, INSERT` on `admin_audit_log` (no UPDATE/DELETE); keep existing CRUD on `users`/`auth_sessions`.

## 6. API security checklist (must-haves)

1. Admin check helper: `def _require_admin(request) -> user` → 401 if no session, 403 if `role != 'admin'` or `disabled_at IS NOT NULL`; also enforced inside `_current_user` join (`AND u.disabled_at IS NULL`).
2. Keep existing CSRF (`_check_csrf`) + same-origin (`_require_same_origin`) on all state-changing admin routes; `GET` admin routes require session but not CSRF (same as `/me`).
3. Generic error messages on auth failures (no user enumeration beyond what exists); 404 vs 403: return 404 for unknown user ids to non-admins, 404 to admins too (no existence oracle change), never leak hashes/tokens.
4. Rate-limit admin reads separately so a table scan can't starve login.
5. `docs_url=None, redoc_url=None` stays; no OpenAPI exposure.
6. Tests in `tests/test_admin_api.py` (extend `FakeDB`): role gate, self-demote/self-disable blocked, disabled login blocked, session revoke works, audit row written per mutation, `me` exposes role.

## 7. Bootstrap & local run

```powershell
# 1. migrate (prompts for schema admin creds, same pattern as initialize_auth_db.py)
.\.venv\Scripts\python.exe scripts\migrate_auth_db.py
# 2. create/promote first admin (password never logged)
.\.venv\Scripts\python.exe scripts\create_admin.py --email you@example.com
# 3. start services as usual (auth :8010 single worker, prediction :8000, shadow runner)
.\.venv\Scripts\python.exe -m uvicorn src.auth.app:app --host 127.0.0.1 --port 8010 --workers 1
.\.venv\Scripts\python.exe -m src.prediction.server
.\.venv\Scripts\python.exe scripts/run_shadow.py
# 4. website
Set-Location website; pnpm install; pnpm dev   # open /admin while signed in as admin
```

Acceptance: sign in as admin → `/admin` loads users table; demote/disable self is rejected; disabling a test user kills their session within one `/me` poll; every action appears in Audit tab.

## 8. Milestones & effort (small, sequential)

| # | Deliverable | Done when |
|---|---|---|
| M0 | migration + `create_admin.py` + `check_schema` + tests | `pytest tests/test_admin_api.py tests/test_auth_api.py` green; fresh + existing DBs migrate cleanly |
| M1 | admin users/sessions/audit API + `/admin` Users/Sessions/Audit UI | admin can search, disable/enable, revoke, paginate audit; non-admin gets 403/redirect |
| M2 | system status/shadow/config/refresh proxy + Ops tab | status matches manual `/health` + cache checks; refresh path reuses single-flight lock and is audited |
| M3 | docs (`ACCOUNT_AUTH.md`, `STARTING_THE_APP.md` admin section) + GRANT notes + hardening tests | another dev can bootstrap from docs alone |

Suggested build order: M0 → M1 API → M1 UI → M2 → M3. Do not start M2 until M1 authz tests pass.

## 9. Risks / decisions needed

1. **Prediction server has no auth** — M2 proxies through auth service instead of securing `:8000`. If admin must work across hosts, revisit (reverse proxy or token). Recommended: keep localhost-only, same as today.
2. **Single-worker limiter** — admin adds load to the same process; keep admin read limits tight and document edge limiting for production.
3. **Last-admin lockout** — enforce in SQL transaction (count active admins > 1 before demote/disable). Owner to confirm this guard vs manual DB recovery.
4. **No hard delete** — disabled accounts retained for audit. Confirm retention expectation.
