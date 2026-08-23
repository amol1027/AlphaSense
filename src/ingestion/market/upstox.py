from datetime import date
import os
import time

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

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


def fetch_historical_candles(
    instrument_key: str,
    from_date: date,
    to_date: date,
    interval_minutes: int = 15,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch historical candles from Upstox V3.

    Retries transient rate-limit/server errors with
    exponential backoff.
    """

    if not 1 <= interval_minutes <= 15:
        raise ValueError(
            "interval_minutes must be between 1 and 15."
        )

    if to_date < from_date:
        raise ValueError(
            "to_date must be greater than or equal to from_date."
        )

    if max_retries < 0:
        raise ValueError(
            "max_retries must be non-negative."
        )

    if retry_delay_seconds < 0:
        raise ValueError(
            "retry_delay_seconds must be non-negative."
        )

    access_token = os.getenv("UPSTOX_ACCESS_TOKEN")

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

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    response = None

    for attempt in range(max_retries + 1):
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
        )

        if response.status_code not in RETRYABLE_STATUS_CODES:
            break

        if attempt == max_retries:
            break

        delay = retry_delay_seconds * (2 ** attempt)
        time.sleep(delay)

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

    return (
        df[OUTPUT_COLUMNS]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )