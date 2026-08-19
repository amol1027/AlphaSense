from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.news.marketaux import MarketauxClient
from src.ingestion.news.gdelt import GDELTNewsClient
from src.ingestion.schemas import NewsArticle


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


def deduplicate(
    articles: list[NewsArticle],
) -> list[NewsArticle]:
    seen_urls = set()
    seen_headlines = set()

    unique = []

    for article in articles:
        url = str(article.url).strip()
        headline = article.headline.strip().lower()

        if url and url in seen_urls:
            continue

        if headline and headline in seen_headlines:
            continue

        if url:
            seen_urls.add(url)

        if headline:
            seen_headlines.add(headline)

        unique.append(article)

    return unique


def collect_marketaux(
    client: MarketauxClient,
    asset: str,
) -> list[NewsArticle]:

    print(f"\nMarketaux: {asset}")

    articles = []

    current = START_DATE

    while current < END_DATE:
        window_end = min(
            current + timedelta(days=7),
            END_DATE,
        )

        print(
            "  Window:",
            current.isoformat(),
            "→",
            window_end.isoformat(),
        )

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

        current = window_end

    return articles


def collect_gdelt(
    client: GDELTNewsClient,
    asset: str,
) -> list[NewsArticle]:
    """
    Collect GDELT articles in small historical windows.

    Windows that hit the GDELT record limit are split
    into smaller windows so we don't silently lose data.
    """

    print(f"\nGDELT: {asset}")

    articles = []

    def collect_window(
        start: datetime,
        end: datetime,
    ) -> None:
        print(
            "  Window:",
            start.isoformat(),
            "→",
            end.isoformat(),
        )

        time.sleep(5)

        batch = client.fetch_news(
            asset=asset,
            max_records=250,
            start_datetime=start,
            end_datetime=end,
        )

        print(
            f"    {len(batch)} articles"
        )

        if len(batch) < 250:
            articles.extend(batch)
            return

        # Exactly 250 means the result may be truncated.
        # Split the window before accepting the result.
        duration = end - start

        if duration <= timedelta(hours=1):
            print(
                "    WARNING: 250 articles in "
                "a <=1 hour window; accepting "
                "the capped result."
            )
            articles.extend(batch)
            return

        midpoint = start + (
            duration / 2
        )

        print(
            "    Hit 250-record limit; "
            "splitting window."
        )

        collect_window(
            start,
            midpoint,
        )

        collect_window(
            midpoint,
            end,
        )

    current = START_DATE

    while current < END_DATE:
        window_end = min(
            current + timedelta(days=1),
            END_DATE,
        )

        try:
            collect_window(
                current,
                window_end,
            )
        except Exception as exc:
            print(
                f"    Window failed: "
                f"{type(exc).__name__}: {exc}"
            )

        current = window_end

    return articles

def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    marketaux = MarketauxClient()
    gdelt = GDELTNewsClient(
        max_retries=3,
        backoff_seconds=5.0,
    )

    all_articles = []

    for asset in ASSETS:
        try:
            all_articles.extend(
                collect_marketaux(
                    marketaux,
                    asset,
                )
            )
        except Exception as exc:
            print(
                f"Marketaux failed for {asset}: "
                f"{type(exc).__name__}: {exc}"
            )

        try:
            all_articles.extend(
                collect_gdelt(
                    gdelt,
                    asset,
                )
            )
        except Exception as exc:
            print(
                f"GDELT failed for {asset}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(
        f"\nRaw articles collected: "
        f"{len(all_articles)}"
    )

    all_articles = deduplicate(
        all_articles
    )
    all_articles = [
    article
    for article in all_articles
    if START_DATE
    <= article.published_at
    < END_DATE
    ]

    print(
        f"After deduplication: "
        f"{len(all_articles)}"
    )

    rows = [
        article_to_row(article)
        for article in all_articles
    ]

    if not rows:
        print(
            "No articles collected. "
            "No output file written."
        )
        return

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
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        "\nArticles by asset:"
    )
    print(
        df["asset"]
        .value_counts()
        .to_string()
    )

    print(
        "\nArticles by source:"
    )
    print(
        df["source"]
        .str.split(":")
        .str[0]
        .value_counts()
        .to_string()
    )

    print(
        "\nDate range:"
    )
    print(
        df["published_at"].min(),
        "→",
        df["published_at"].max(),
    )


if __name__ == "__main__":
    main()