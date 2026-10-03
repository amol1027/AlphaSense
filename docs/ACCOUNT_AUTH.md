# AlphaSense account service

The public research site now has registration and sign-in pages. Authentication is a separate FastAPI service; it does not change the prediction API or make the research pages private.

## Local MySQL setup

MySQL Server must be running. In a MySQL administrator session, create a dedicated database and a limited application user (choose a private password locally):

```sql
CREATE DATABASE alphasense_auth
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE USER 'alphasense_app'@'127.0.0.1' IDENTIFIED BY 'choose-a-local-password';
GRANT SELECT, INSERT, UPDATE, DELETE ON alphasense_auth.*
  TO 'alphasense_app'@'127.0.0.1';
```

Copy the MySQL settings from [`.env.example`](../.env.example) into the existing root `.env`. Keep the app user's password in `.env`; do not replace or share any existing provider tokens there. The app user only needs CRUD access. Table creation uses a separate administrator credential entered privately by the setup script.

From the repository root, install the added Python dependencies, then create the tables:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\initialize_auth_db.py
```

The initializer asks for the MySQL schema administrator username and password without echoing the password. It creates the `users` and `auth_sessions` tables in `MYSQL_DATABASE`. It does not store the administrator credentials.

## Start the API and site

In one PowerShell terminal, start the account API (single worker is required — the login/register rate limiter is per-process memory, so extra workers would each enforce an independent budget):

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.auth.app:app --host 127.0.0.1 --port 8010 --workers 1
```

Password policy: registration requires at least 10 characters and rejects a small denylist of common passwords; sign-in accepts any non-empty password so accounts created under the older 6-character policy keep working. The attempt limiter defaults to 10 tries per 10 minutes per IP and is tunable via `AUTH_RATE_LIMIT` / `AUTH_RATE_WINDOW_SECONDS`; exceeded callers get HTTP 429 with a `Retry-After` header. `GET /api/auth/health` returns 200 when MySQL is reachable, 503 otherwise.

In another terminal, configure and start the React site:

```powershell
Copy-Item website\.env.example website\.env.local
Set-Location website
pnpm install
pnpm dev
```

Open the Vite URL (normally `http://127.0.0.1:5173`). Set `VITE_AUTH_API_BASE_URL` in `website/.env.local` if the API uses another URL. The API accepts the default Vite dev and local preview origins; add any custom origin to `AUTH_ALLOWED_ORIGINS` in the root `.env` and restart the API.

## Production configuration

Serve the API only over HTTPS. Set `AUTH_COOKIE_SECURE=true`; the service refuses to boot with an `https://` origin in `AUTH_ALLOWED_ORIGINS` unless it is set. If the frontend and API are on separate sites, set `AUTH_COOKIE_SAMESITE=none` and use a browser origin allowlist containing only the exact production frontend origins. A same-site reverse proxy is preferable because it avoids third-party-cookie restrictions. Never use `*` as an allowed origin with account cookies.

Set `VITE_AUTH_API_BASE_URL` to the deployed API URL and configure the static host to rewrite `/login`, `/register`, and `/admin` to the site's `index.html` (SPA fallback).

## Administration (M0+M1)

Existing databases need the one-time migration before admin works:

```powershell
.\.venv\Scripts\python.exe scripts\migrate_auth_db.py
.\.venv\Scripts\python.exe scripts\create_admin.py --email you@example.com
```

This adds `users.role`, `users.disabled_at`, and the append-only `admin_audit_log` table. Sign in as the admin and open `/admin` for Users (search, promote/demote, disable/enable, revoke sessions), Sessions, Audit log, and Ops. Admin API lives under `/api/admin/*` on the account service (`:8010`); the prediction service (`:8000`) stays localhost-only with no auth, so cross-host admin must go through a reverse proxy, not direct `:8000` exposure. Guards: admins cannot change their own role, disable themselves, revoke their own session (use sign out), or demote/disable the last active admin. Disabling a user blocks login and `/me` immediately and deletes their sessions.

The Ops tab aggregates what operators otherwise check by hand: prediction `/health` (probed server-side, never from the browser), account-DB reachability, market/news cache row counts and refresh timestamps, shadow-log scoring counts, per-asset model-artifact presence, Upstox token presence (boolean only — the value is never returned), and website-build presence, plus a non-secret configuration summary. Its "Fetch latest data" button runs the same flow as the market service's refresh (candles + outcome labels + shadow log) under a single-flight lock and writes a `system.refresh` audit entry. Model retraining stays a CLI action (`scripts/train_volatility_model.py`); the admin UI cannot upload or retrain artifacts.

The included attempt limiter is per process, suitable for local development. A public deployment should also apply rate limits at its edge and use a production MySQL account, TLS connection, backups, and secret manager. Password reset, email verification, and account deletion are not part of this initial implementation.

Passwords are hashed with Argon2id; session bearer tokens are stored as SHA-256 digests and expire after seven days. See the [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html) and the [MySQL Connector/Python guide](https://dev.mysql.com/doc/connector-python/en/).
