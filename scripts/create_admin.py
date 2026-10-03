"""Create or promote the first AlphaSense admin.

Usage:
    python scripts/create_admin.py --email you@example.com [--name "Your Name"]

- Prompts for a password (min 6 chars) only when creating a new user.
- Promoting an existing user never asks for or changes their password.
- Refuses to run when the users table has no role/disabled_at columns
  (run scripts/migrate_auth_db.py first).
"""

import argparse
import sys
from getpass import getpass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from argon2 import PasswordHasher
from dotenv import load_dotenv

from src.auth import database

load_dotenv()
HASHER = PasswordHasher(time_cost=2, memory_cost=19_456, parallelism=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or promote an AlphaSense admin")
    parser.add_argument("--email", required=True, help="Account email to create/promote")
    parser.add_argument("--name", default="Admin", help="Display name for new accounts")
    args = parser.parse_args()
    email = args.email.strip().lower()

    try:
        existing = database.fetch_one(
            "SELECT id, email, display_name, role, disabled_at FROM users WHERE email = %s",
            (email,),
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not read users table ({exc}). Run scripts/migrate_auth_db.py first."
        ) from exc

    if existing is None:
        while True:
            password = getpass("New admin password (min 6 chars, hidden): ")
            if len(password) < 6:
                print("Use at least 6 characters.")
                continue
            confirm = getpass("Confirm password: ")
            if password != confirm:
                print("Passwords do not match.")
                continue
            break
        user_id = database.execute(
            "INSERT INTO users (email, display_name, password_hash, role) VALUES (%s, %s, %s, 'admin')",
            (email, args.name, HASHER.hash(password)),
        )
        print(f"Created admin {email} (id={user_id}).")
        return

    if existing.get("disabled_at") is not None:
        raise SystemExit(f"Refusing: {email} is disabled; re-enable it before promoting.")
    if existing.get("role") == "admin":
        print(f"{email} is already an admin (id={existing['id']}). Nothing to do.")
        return
    database.execute("UPDATE users SET role = 'admin' WHERE id = %s", (existing["id"],))
    print(f"Promoted {email} (id={existing['id']}) to admin.")


if __name__ == "__main__":
    main()
