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


URL = (
    "https://api.upstox.com/v2/instruments/search"
)


def search_instrument(symbol: str) -> None:
    response = requests.get(
        URL,
        headers={
            "Accept": "application/json",
            "Authorization": (
                f"Bearer {ACCESS_TOKEN}"
            ),
        },
        params={
            "query": symbol,
            "exchanges": "NSE",
            "segments": "EQ",
            "page_number": 1,
            "records": 10,
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    print(f"\n=== {symbol} ===")

    for instrument in payload.get("data", []):
        print(
            "name:",
            instrument.get("name"),
        )
        print(
            "trading_symbol:",
            instrument.get(
                "trading_symbol"
            ),
        )
        print(
            "exchange:",
            instrument.get("exchange"),
        )
        print(
            "segment:",
            instrument.get("segment"),
        )
        print(
            "instrument_type:",
            instrument.get(
                "instrument_type"
            ),
        )
        print(
            "instrument_key:",
            instrument.get(
                "instrument_key"
            ),
        )
        print()


def main():
    search_instrument("TCS")
    search_instrument("RELIANCE")


if __name__ == "__main__":
    main()