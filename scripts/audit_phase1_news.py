from pathlib import Path

import pandas as pd

from src.ingestion.news.audit import audit_news_data


INPUT_PATH = Path(
    "data/raw/news/research_news.csv"
)

START_DATE = pd.Timestamp(
    "2026-07-01",
    tz="UTC",
)

END_DATE = pd.Timestamp(
    "2026-08-11",
    tz="UTC",
)


def main() -> None:
    print("=" * 70)
    print("PHASE 1 NEWS DATA AUDIT")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"News dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Input: {INPUT_PATH}"
    )
    print()

    result = audit_news_data(
        df,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print(
        f"Rows:                       {result.rows:,}"
    )

    print(
        f"Date range:                 "
        f"{result.start_timestamp} -> "
        f"{result.end_timestamp}"
    )

    print()

    print("DATA QUALITY")
    print("-" * 70)

    print(
        f"Invalid timestamps:         "
        f"{result.invalid_timestamps}"
    )

    print(
        f"Null timestamps:            "
        f"{result.null_timestamps}"
    )

    print(
        f"Null headlines:             "
        f"{result.null_headlines}"
    )

    print(
        f"Empty headlines:            "
        f"{result.empty_headlines}"
    )

    print(
        f"Empty article bodies:       "
        f"{result.empty_bodies}"
    )

    print(
        f"Duplicate URLs:             "
        f"{result.duplicate_urls}"
    )

    print(
        f"Duplicate normalized "
        f"headlines:                 "
        f"{result.duplicate_headlines}"
    )

    print(
        f"Invalid assets:             "
        f"{result.invalid_assets}"
    )

    print(
        f"Invalid exchanges:          "
        f"{result.invalid_exchanges}"
    )

    print(
        f"Future timestamps:          "
        f"{result.future_timestamps}"
    )

    print(
        f"Out-of-window timestamps:   "
        f"{result.out_of_window_timestamps}"
    )

    print()

    print("ASSETS")
    print("-" * 70)

    for asset in sorted(result.assets):
        count = int(
            (
                df["asset"]
                == asset
            ).sum()
        )

        print(
            f"{asset:<20} {count:>8,}"
        )

    print()

    print("PROVIDERS")
    print("-" * 70)

    for provider, count in sorted(
        result.sources.items()
    ):
        print(
            f"{provider:<20} {count:>8,}"
        )

    print()

    print("OVERALL STATUS:", result.status)

    print("=" * 70)

    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()