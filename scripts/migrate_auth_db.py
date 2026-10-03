"""Apply pending src/auth/migrations/*.sql to the configured MySQL database.

Idempotent: each statement is skipped when INFORMATION_SCHEMA shows it is
already applied (columns/keys/tables). Prompts for a schema administrator
credential like initialize_auth_db.py; never stores it.
"""

from getpass import getpass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

import mysql.connector


def _column_exists(cursor, table: str, column: str) -> bool:
    # fetchall (not fetchone): INFORMATION_SCHEMA queries can return several
    # rows, and leaving any row unread on an unbuffered cursor makes the next
    # execute() fail with "Unread result found".
    cursor.execute(
        """SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
           WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s""",
        (table, column),
    )
    return len(cursor.fetchall()) > 0


def _index_exists(cursor, table: str, index: str) -> bool:
    cursor.execute(
        """SELECT 1 FROM INFORMATION_SCHEMA.STATISTICS
           WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND INDEX_NAME = %s""",
        (table, index),
    )
    return len(cursor.fetchall()) > 0


def _table_exists(cursor, table: str) -> bool:
    cursor.execute(
        """SELECT 1 FROM INFORMATION_SCHEMA.TABLES
           WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s""",
        (table,),
    )
    return len(cursor.fetchall()) > 0


def apply_001(cursor) -> None:
    if not _column_exists(cursor, "users", "role"):
        cursor.execute("ALTER TABLE users ADD COLUMN role ENUM('user','admin') NOT NULL DEFAULT 'user'")
        print("added users.role")
    if not _column_exists(cursor, "users", "disabled_at"):
        cursor.execute("ALTER TABLE users ADD COLUMN disabled_at DATETIME NULL")
        print("added users.disabled_at")
    if not _index_exists(cursor, "users", "ix_users_role_disabled"):
        cursor.execute("ALTER TABLE users ADD KEY ix_users_role_disabled (role, disabled_at)")
        print("added ix_users_role_disabled")
    if not _table_exists(cursor, "admin_audit_log"):
        cursor.execute(
            """CREATE TABLE admin_audit_log (
              id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
              actor_id BIGINT UNSIGNED NOT NULL,
              action VARCHAR(64) NOT NULL,
              target_user_id BIGINT UNSIGNED NULL,
              detail JSON NULL,
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (id),
              KEY ix_audit_actor (actor_id),
              KEY ix_audit_created (created_at),
              CONSTRAINT fk_audit_actor FOREIGN KEY (actor_id)
                REFERENCES users (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"""
        )
        print("created admin_audit_log")


def apply_002(cursor) -> None:
    if not _column_exists(cursor, "users", "whatsapp_e164"):
        cursor.execute("ALTER TABLE users ADD COLUMN whatsapp_e164 VARCHAR(20) NULL")
        print("added users.whatsapp_e164")
    if not _table_exists(cursor, "notification_prefs"):
        cursor.execute(
            """CREATE TABLE notification_prefs (
              user_id BIGINT UNSIGNED NOT NULL,
              email_enabled TINYINT(1) NOT NULL DEFAULT 1,
              whatsapp_enabled TINYINT(1) NOT NULL DEFAULT 0,
              assets JSON NULL,
              min_probability DECIMAL(4,3) NOT NULL DEFAULT 0.700,
              daily_summary TINYINT(1) NOT NULL DEFAULT 0,
              updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (user_id),
              CONSTRAINT fk_notif_prefs_user FOREIGN KEY (user_id)
                REFERENCES users (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"""
        )
        print("created notification_prefs")
    if not _table_exists(cursor, "notification_log"):
        cursor.execute(
            """CREATE TABLE notification_log (
              id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
              user_id BIGINT UNSIGNED NOT NULL,
              channel ENUM('email','whatsapp') NOT NULL,
              type VARCHAR(32) NOT NULL DEFAULT 'high_range',
              asset VARCHAR(16) NOT NULL,
              dedupe_key VARCHAR(128) NOT NULL,
              status VARCHAR(16) NOT NULL DEFAULT 'sent',
              provider_msg_id VARCHAR(128) NULL,
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (id),
              UNIQUE KEY uq_notif_dedupe (dedupe_key, channel),
              KEY ix_notif_user_created (user_id, created_at),
              CONSTRAINT fk_notif_log_user FOREIGN KEY (user_id)
                REFERENCES users (id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"""
        )
        print("created notification_log")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    missing = [name for name in ("MYSQL_HOST", "MYSQL_DATABASE", "MYSQL_USER") if not os.getenv(name)]
    if missing:
        raise SystemExit(
            f"Missing MySQL configuration in .env: {', '.join(missing)}. Copy them from .env.example."
        )
    user = input("MySQL schema administrator user: ").strip()
    password = getpass("MySQL schema administrator password: ")
    try:
        connection = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=os.environ["MYSQL_DATABASE"],
            user=user,
            password=password,
            connection_timeout=5,
        )
    except mysql.connector.Error as exc:
        raise SystemExit(f"Could not connect to MySQL ({exc}). Is the server running?") from exc
    try:
        # Buffered: existence probes must never leave unread rows behind
        # (see _column_exists), otherwise later statements fail.
        cursor = connection.cursor(buffered=True)
        try:
            if not _table_exists(cursor, "users"):
                raise SystemExit(
                    "The users table does not exist. Run scripts/initialize_auth_db.py first."
                )
            apply_001(cursor)
            apply_002(cursor)
            connection.commit()
        finally:
            cursor.close()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    print("Auth migrations are up to date.")
