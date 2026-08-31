from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta
import json
import time

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.news.marketaux import MarketauxClient
from src.ingestion.news.gdelt import GDELTNewsClient
from src.ingestion.schemas import NewsArticle
from src.ingestion.news.dedup import deduplicate_news


START_DATE = datetime(
    2026,
    7,
    1,
    tzinfo=timezone.utc,
)

END_DATE = datetime(
    2026,
    8,
    11,
    tzinfo=timezone.utc,
)

ASSETS = [
    "TCS",
    "RELIANCE",
]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "news"
    / "research_news.csv"
)

TEMP_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "news"
    / "research_news_expanded.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "news"
    / "news_collection_checkpoint.json"
)

FAILED_WINDOWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "news"
    / "news_failed_windows.json"
)


# ============================================================
# Helpers
# ============================================================

def article_to_row(article: NewsArticle) -> dict:
    return {
        "asset": article.asset,
        "exchange": article.exchange,
        "published_at": article.published_at,
        "source": article.source,
        "headline": article.headline,
        "text": article.text,
        "url": str(article.url),
    }


def save_checkpoint(
    completed_windows: set[str],
) -> None:
    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "completed_windows": sorted(
            completed_windows
        )
    }

    CHECKPOINT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_checkpoint() -> set[str]:
    if not CHECKPOINT_PATH.exists():
        return set()

    try:
        payload = json.loads(
            CHECKPOINT_PATH.read_text(
                encoding="utf-8"
            )
        )

        return set(
            payload.get(
                "completed_windows",
                [],
            )
        )

    except Exception:
        print(
            "WARNING: checkpoint file could not "
            "be read. Starting without checkpoint."
        )

        return set()


def record_failed_window(
    provider: str,
    asset: str,
    start: datetime,
    end: datetime,
    error: Exception,
) -> None:

    FAILED_WINDOWS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if FAILED_WINDOWS_PATH.exists():
        try:
            failures = json.loads(
                FAILED_WINDOWS_PATH.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            failures = []
    else:
        failures = []

    failures.append(
        {
            "provider": provider,
            "asset": asset,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )

    FAILED_WINDOWS_PATH.write_text(
        json.dumps(
            failures,
            indent=2,
        ),
        encoding="utf-8",
    )


def window_key(
    provider: str,
    asset: str,
    start: datetime,
    end: datetime,
) -> str:
    return (
        f"{provider}|"
        f"{asset}|"
        f"{start.isoformat()}|"
        f"{end.isoformat()}"
    )


def save_articles(
    articles: list[NewsArticle],
    path: Path,
) -> None:

    if not articles:
        return

    rows = [
        article_to_row(article)
        for article in articles
    ]

    df = pd.DataFrame(rows)

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        utc=True,
    )

    df = df.sort_values(
        [
            "asset",
            "published_at",
        ]
    ).reset_index(drop=True)

    df.to_csv(
        path,
        index=False,
    )


# ============================================================
# Marketaux
# ============================================================

def collect_marketaux(
    client: MarketauxClient,
    asset: str,
    completed_windows: set[str],
) -> list[NewsArticle]:

    print(f"\nMarketaux: {asset}")

    articles = []

    current = START_DATE

    while current < END_DATE:

        window_end = min(
            current + timedelta(days=7),
            END_DATE,
        )

        key = window_key(
            "marketaux",
            asset,
            current,
            window_end,
        )

        if key in completed_windows:
            print(
                "  SKIP completed window:",
                current.isoformat(),
                "→",
                window_end.isoformat(),
            )

            current = window_end
            continue

        print(
            "  Window:",
            current.isoformat(),
            "→",
            window_end.isoformat(),
        )

        try:

            page = 1

            while True:

                batch = client.fetch_news(
                    asset=asset,
                    limit=50,
                    published_after=current.isoformat(),
                    published_before=window_end.isoformat(),
                    page=page,
                )

                print(
                    f"    page {page}: "
                    f"{len(batch)} articles"
                )

                articles.extend(batch)

                if len(batch) < 50:
                    break

                page += 1

            completed_windows.add(key)
            save_checkpoint(
                completed_windows
            )

        except (
            requests.RequestException,
            TimeoutError,
            Exception,
        ) as exc:

            print(
                f"    FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

            record_failed_window(
                "marketaux",
                asset,
                current,
                window_end,
                exc,
            )

        current = window_end

    return articles


# ============================================================
# GDELT
# ============================================================

def collect_gdelt(
    client: GDELTNewsClient,
    asset: str,
    completed_windows: set[str],
) -> list[NewsArticle]:

    print(f"\nGDELT: {asset}")

    articles = []

    current = START_DATE

    while current < END_DATE:

        window_end = min(
            current + timedelta(hours=6),
            END_DATE,
        )

        key = window_key(
            "gdelt",
            asset,
            current,
            window_end,
        )

        if key in completed_windows:

            print(
                "  SKIP completed window:",
                current.isoformat(),
                "→",
                window_end.isoformat(),
            )

            current = window_end
            continue

        print(
            "  Window:",
            current.isoformat(),
            "→",
            window_end.isoformat(),
        )

        try:

            # Small pause prevents aggressive request bursts.
            time.sleep(2)

            batch = client.fetch_news(
                asset=asset,
                max_records=250,
                start_datetime=current,
                end_datetime=window_end,
            )

            print(
                f"    {len(batch)} articles"
            )

            articles.extend(batch)

            completed_windows.add(key)

            save_checkpoint(
                completed_windows
            )

        except (
            requests.RequestException,
            TimeoutError,
            Exception,
        ) as exc:

            print(
                f"    FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

            record_failed_window(
                "gdelt",
                asset,
                current,
                window_end,
                exc,
            )

            # Do not mark the window complete.
            # It can be retried later.

        current = window_end

    return articles


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_windows = load_checkpoint()

    print("=" * 80)
    print("HISTORICAL NEWS COLLECTION")
    print("=" * 80)

    print()
    print(
        f"Collection period: "
        f"{START_DATE.isoformat()} → "
        f"{END_DATE.isoformat()}"
    )

    print(
        f"Existing output: {OUTPUT_PATH}"
    )

    print(
        f"Expanded output: {TEMP_OUTPUT_PATH}"
    )

    print(
        f"Completed windows in checkpoint: "
        f"{len(completed_windows)}"
    )

    # --------------------------------------------------------
    # Existing data
    # --------------------------------------------------------

    existing_articles = []

    if OUTPUT_PATH.exists():

        print()
        print(
            "Loading existing news dataset..."
        )

        existing_df = pd.read_csv(
            OUTPUT_PATH
        )

        existing_df["published_at"] = (
            pd.to_datetime(
                existing_df["published_at"],
                utc=True,
            )
        )

        for row in existing_df.itertuples(
            index=False
        ):

            existing_articles.append(
                NewsArticle(
                    asset=row.asset,
                    exchange=row.exchange,
                    published_at=row.published_at.to_pydatetime(),
                    source=row.source,
                    headline=row.headline,
                    text=(
                        ""
                        if pd.isna(row.text)
                        else str(row.text)
                    ),
                    url=row.url,
                )
            )

        print(
            f"Existing articles: "
            f"{len(existing_articles):,}"
        )

    # --------------------------------------------------------
    # Providers
    # --------------------------------------------------------

    marketaux = MarketauxClient()

    gdelt = GDELTNewsClient(
        timeout=30,
        max_retries=3,
        backoff_seconds=5.0,
    )

    collected_articles = []

    # --------------------------------------------------------
    # Collection
    # --------------------------------------------------------

    for asset in ASSETS:

        try:

            collected_articles.extend(
                collect_marketaux(
                    marketaux,
                    asset,
                    completed_windows,
                )
            )

        except Exception as exc:

            print(
                f"Marketaux failed for {asset}: "
                f"{type(exc).__name__}: {exc}"
            )

        try:

            collected_articles.extend(
                collect_gdelt(
                    gdelt,
                    asset,
                    completed_windows,
                )
            )

        except Exception as exc:

            print(
                f"GDELT failed for {asset}: "
                f"{type(exc).__name__}: {exc}"
            )

    print()
    print(
        f"New articles collected: "
        f"{len(collected_articles):,}"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    all_articles = (
        existing_articles
        + collected_articles
    )

    print(
        f"Combined articles before deduplication: "
        f"{len(all_articles):,}"
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    deduplication_result = (
        deduplicate_news(
            all_articles
        )
    )

    all_articles = (
        deduplication_result.articles
    )

    print(
        f"Duplicates removed: "
        f"{deduplication_result.duplicate_count:,}"
    )

    # --------------------------------------------------------
    # Date validation
    # --------------------------------------------------------

    all_articles = [
        article
        for article in all_articles
        if START_DATE
        <= article.published_at
        < END_DATE
    ]

    print(
        f"Articles inside requested date range: "
        f"{len(all_articles):,}"
    )

    # --------------------------------------------------------
    # Save expanded dataset
    # --------------------------------------------------------

    if not all_articles:

        print(
            "\nNo articles available."
        )

        print(
            "Existing dataset was NOT modified."
        )

        return

    save_articles(
        all_articles,
        TEMP_OUTPUT_PATH,
    )

    print()
    print(
        f"Saved expanded dataset:"
    )
    print(
        TEMP_OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    final_df = pd.read_csv(
        TEMP_OUTPUT_PATH
    )

    final_df["published_at"] = (
        pd.to_datetime(
            final_df["published_at"],
            utc=True,
        )
    )

    print()
    print(
        "ARTICLES BY ASSET"
    )
    print("-" * 80)

    print(
        final_df[
            "asset"
        ].value_counts()
        .to_string()
    )

    print()
    print(
        "ARTICLES BY SOURCE"
    )
    print("-" * 80)

    print(
        final_df[
            "source"
        ]
        .str.split(":")
        .str[0]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "DATE RANGE"
    )
    print("-" * 80)

    print(
        final_df["published_at"].min(),
        "→",
        final_df["published_at"].max(),
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The existing research_news.csv was not overwritten."
    )

    print(
        "Review research_news_expanded.csv before promotion."
    )

    print("=" * 80)
    print(
        "HISTORICAL NEWS COLLECTION COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()