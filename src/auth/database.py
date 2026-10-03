"""MySQL connections and schema support for account storage."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import mysql.connector
from dotenv import load_dotenv
from mysql.connector.pooling import MySQLConnectionPool


load_dotenv()


@lru_cache(maxsize=1)
def get_pool() -> MySQLConnectionPool:
    required = ("MYSQL_HOST", "MYSQL_DATABASE", "MYSQL_USER")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing MySQL configuration: {', '.join(missing)}")

    return MySQLConnectionPool(
        pool_name="alphasense_auth",
        pool_size=5,
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.environ["MYSQL_DATABASE"],
        user=os.environ["MYSQL_USER"],
        password=os.getenv("MYSQL_PASSWORD", ""),
        connection_timeout=5,
        autocommit=False,
    )


def get_connection():
    return get_pool().get_connection()


def initialize_schema() -> None:
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    schema_user = os.getenv("MYSQL_SCHEMA_USER")
    schema_password = os.getenv("MYSQL_SCHEMA_PASSWORD")
    if not schema_user or schema_password is None:
        raise RuntimeError("Set MYSQL_SCHEMA_USER and MYSQL_SCHEMA_PASSWORD for the one-time schema setup")
    connection = mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.environ["MYSQL_DATABASE"],
        user=schema_user,
        password=schema_password,
        connection_timeout=5,
    )
    try:
        cursor = connection.cursor()
        try:
            for statement in schema.split(";"):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
            connection.commit()
        finally:
            cursor.close()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def check_schema() -> None:
    fetch_one("SELECT id FROM users LIMIT 0")
    fetch_one("SELECT token_hash FROM auth_sessions LIMIT 0")
    execute("DELETE FROM auth_sessions WHERE expires_at <= UTC_TIMESTAMP()")
    # M0 admin columns are optional at runtime so older DBs keep booting
    # before migrate_auth_db.py is run; role defaults to "user" in app code.
    try:
        fetch_one("SELECT role FROM users LIMIT 0")
        fetch_one("SELECT id FROM admin_audit_log LIMIT 0")
    except Exception:
        pass


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    connection = get_connection()
    try:
        # Buffered so a caller that ignores trailing rows can never leave
        # unread results behind (mysql-connector raises "Unread result found"
        # on the next use of such a cursor).
        cursor = connection.cursor(dictionary=True, buffered=True)
        try:
            cursor.execute(query, params)
            return list(cursor.fetchall())
        finally:
            cursor.close()
    finally:
        connection.close()


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    connection = get_connection()
    try:
        cursor = connection.cursor(dictionary=True, buffered=True)
        try:
            cursor.execute(query, params)
            return cursor.fetchone()
        finally:
            cursor.close()
    finally:
        connection.close()


def execute(query: str, params: tuple[Any, ...] = ()) -> int:
    connection = get_connection()
    try:
        cursor = connection.cursor(buffered=True)
        try:
            cursor.execute(query, params)
            connection.commit()
            return int(cursor.lastrowid or 0)
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
    finally:
        connection.close()
