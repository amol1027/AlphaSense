"""Create the AlphaSense account tables in the configured MySQL database."""

from getpass import getpass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auth.database import initialize_schema


if __name__ == "__main__":
    import os

    os.environ["MYSQL_SCHEMA_USER"] = input("MySQL schema administrator user: ").strip()
    os.environ["MYSQL_SCHEMA_PASSWORD"] = getpass("MySQL schema administrator password: ")
    initialize_schema()
    print("AlphaSense account tables are ready.")
