from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datetime import date

from src.ingestion.market.downloader import (
    download_market_data,
)


INSTRUMENTS = {
    "TCS": {
        "instrument_key": "NSE_EQ|INE467B01029",
        "exchange": "NSE",
    },
    "RELIANCE": {
        "instrument_key": "NSE_EQ|INE002A01018",
        "exchange": "NSE",
    },
}

FROM_DATE = date(2026, 8, 1)
TO_DATE = date(2026, 8, 10)


def main():
    for symbol, config in INSTRUMENTS.items():
        output_path = (
        f"data/raw/market/"
        f"{symbol.lower()}_15m.csv"
    )

    df = download_market_data(
        instrument_key=config["instrument_key"],
        asset=symbol,
        exchange=config["exchange"],
        from_date=FROM_DATE,
        to_date=TO_DATE,
        output_path=output_path,
        interval_minutes=15,
    )

    print(
        f"{symbol}: {len(df)} candles"
    )

    print(
        f"  First: {df['timestamp'].min()}"
    )

    print(
        f"  Last:  {df['timestamp'].max()}"
    )


if __name__ == "__main__":
    main()