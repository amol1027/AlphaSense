from datetime import date

import os

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()


BASE_URL = "https://api.upstox.com/v3"

OUTPUT_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def fetch_historical_candles(
    instrument_key: str,
    from_date: date,
    to_date: date,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """
    Fetch historical candles from Upstox V3.

    Returns candles in the canonical market schema:

        timestamp
        open
        high
        low
        close
        volume

    The returned DataFrame is sorted chronologically.
    """

    if not 1 <= interval_minutes <= 15:
        raise ValueError(
            "interval_minutes must be between 1 and 15."
        )

    if to_date < from_date:
        raise ValueError(
            "to_date must be greater than or equal to from_date."
        )

    access_token = os.getenv(
        "UPSTOX_ACCESS_TOKEN"
    )

    if not access_token:
        raise ValueError(
            "UPSTOX_ACCESS_TOKEN is not set."
        )

    url = (
        f"{BASE_URL}/historical-candle/"
        f"{instrument_key}/minutes/"
        f"{interval_minutes}/"
        f"{to_date.isoformat()}/"
        f"{from_date.isoformat()}"
    )

    response = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": (
                f"Bearer {access_token}"
            ),
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    try:
        candles = payload["data"]["candles"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Unexpected Upstox historical candle response."
        ) from exc

    if not candles:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    rows = []

    for candle in candles:
        if len(candle) < 6:
            raise ValueError(
                "Malformed candle returned by Upstox."
            )

        rows.append(
            {
                "timestamp": candle[0],
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            }
        )

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df = df[
        OUTPUT_COLUMNS
    ].sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df