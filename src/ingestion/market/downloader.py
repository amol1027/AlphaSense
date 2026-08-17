from datetime import date
from pathlib import Path

import pandas as pd

from src.ingestion.market.upstox import (
    fetch_historical_candles,
)
from src.ingestion.market.validation import (
    validate_market_data,
)


OUTPUT_COLUMNS = [
    "asset",
    "exchange",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def download_market_data(
    instrument_key: str,
    asset: str,
    exchange: str,
    from_date: date,
    to_date: date,
    output_path: str | Path,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """
    Download historical market data, attach asset/exchange
    identity, validate it, and save it as CSV.
    """

    if not asset:
        raise ValueError(
            "asset cannot be empty."
        )

    if not exchange:
        raise ValueError(
            "exchange cannot be empty."
        )

    if to_date < from_date:
        raise ValueError(
            "to_date must be greater than or equal to from_date."
        )

    df = fetch_historical_candles(
        instrument_key=instrument_key,
        from_date=from_date,
        to_date=to_date,
        interval_minutes=interval_minutes,
    )

    if df.empty:
        raise ValueError(
            "No market data returned by provider."
        )

    validated = validate_market_data(df)

    validated.insert(
        0,
        "exchange",
        exchange,
    )

    validated.insert(
        0,
        "asset",
        asset,
    )

    validated = validated[
        OUTPUT_COLUMNS
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validated.to_csv(
        output_path,
        index=False,
    )

    return validated