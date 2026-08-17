from datetime import date
import os

import requests
from dotenv import load_dotenv


load_dotenv()

ACCESS_TOKEN = os.getenv(
    "UPSTOX_ACCESS_TOKEN"
)

if not ACCESS_TOKEN:
    raise RuntimeError(
        "UPSTOX_ACCESS_TOKEN is not set."
    )


INSTRUMENT_KEY = (
    "NSE_EQ|INE467B01029"
)

FROM_DATE = date(2026, 8, 10)
TO_DATE = date(2026, 8, 10)


url = (
    "https://api.upstox.com/v3/"
    "historical-candle/"
    f"{INSTRUMENT_KEY}/"
    "minutes/15/"
    f"{TO_DATE.isoformat()}/"
    f"{FROM_DATE.isoformat()}"
)

response = requests.get(
    url,
    headers={
        "Accept": "application/json",
        "Authorization": (
            f"Bearer {ACCESS_TOKEN}"
        ),
    },
    timeout=30,
)

print("HTTP status:", response.status_code)

response.raise_for_status()

payload = response.json()

candles = payload["data"]["candles"]

print("Candle count:", len(candles))

if candles:
    print("\nFirst candle:")
    print(candles[0])

    print("\nLast candle:")
    print(candles[-1])