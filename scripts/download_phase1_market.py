from datetime import date, timedelta
from pathlib import Path

from src.ingestion.market.downloader import download_market_data


START_DATE = date(2024, 1, 1)
END_DATE = date(2026, 8, 21)

OUTPUT_DIR = Path("data/raw/market")

ASSETS = {
    "RELIANCE": {
        "instrument_key": "NSE_EQ|INE002A01018",
        "exchange": "NSE",
    },
    "TCS": {
        "instrument_key": "NSE_EQ|INE467B01029",
        "exchange": "NSE",
    },
}


def latest_completed_trading_date() -> date:
    """
    Return the most recent weekday before today.

    The exchange holiday calendar is handled later during
    validation; this only prevents downloading today's
    potentially incomplete session.
    """
    today = date.today()
    candidate = today - timedelta(days=1)

    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)

    return candidate


def main() -> None:
    end_date = latest_completed_trading_date()

    print("=" * 70)
    print("PHASE 1 MARKET DATA DOWNLOAD")
    print("=" * 70)
    print(f"Start date: {START_DATE}")
    print(f"End date:   {end_date}")
    print("Interval:   15 minutes")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for asset, config in ASSETS.items():
        print("-" * 70)
        print(f"Downloading {asset}...")
        print("-" * 70)

        output_path = (
            OUTPUT_DIR
            / f"phase1_{asset.lower()}_15m.csv"
        )

        df = download_market_data(
            instrument_key=config["instrument_key"],
            asset=asset,
            exchange=config["exchange"],
            from_date=START_DATE,
            to_date=end_date,
            output_path=output_path,
            interval_minutes=15,
            max_days_per_request=31,
        )

        print(f"Saved: {output_path}")
        print(f"Rows:  {len(df):,}")
        print(f"Start: {df['timestamp'].min()}")
        print(f"End:   {df['timestamp'].max()}")
        print(
            f"Duplicates: "
            f"{df['timestamp'].duplicated().sum()}"
        )
        print()


if __name__ == "__main__":
    main()