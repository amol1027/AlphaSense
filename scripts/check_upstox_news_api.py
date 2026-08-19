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


url = "https://api.upstox.com/v2/news"

params = {
    "category": "instrument_keys",
    "instrument_keys": (
        "NSE_EQ|INE467B01029,"
        "NSE_EQ|INE002A01018"
    ),
    "page_number": 1,
    "page_size": 10,
}

response = requests.get(
    url,
    headers={
        "Accept": "application/json",
        "Authorization": (
            f"Bearer {ACCESS_TOKEN}"
        ),
    },
    params=params,
    timeout=30,
)

print("HTTP status:", response.status_code)
print(response.text[:10000])