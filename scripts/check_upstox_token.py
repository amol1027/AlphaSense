"""Upstox token status check (exit 0 = valid, 1 = expired/missing).

Upstox access tokens expire daily and require manual OAuth login to renew,
so this is a fail-loud gate for any refresh job — not an auto-refresh.
Usage: .venv\\Scripts\\python.exe scripts/check_upstox_token.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def token_status() -> tuple[bool, str]:
    """Probe the Upstox token. Returns (ok, message); importable so runners
    can gate on it instead of shelling out to this script."""
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if not token:
        return False, "UPSTOX_ACCESS_TOKEN is not set."
    try:
        response = requests.get(
            "https://api.upstox.com/v3/historical-candle/"
            "NSE_EQ|INE467B01029/minutes/15/"
            "2026-09-25/2026-09-25",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        return False, f"Token check request failed: {exc}"
    if response.status_code == 200:
        return True, "Upstox token: VALID"
    if response.status_code == 401:
        return False, "Upstox token: EXPIRED/INVALID (manual OAuth login required)."
    return False, f"Upstox token: unexpected HTTP {response.status_code}."


def main() -> int:
    ok, message = token_status()
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
