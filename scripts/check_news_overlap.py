import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
    ),
)

from src.ingestion.news.marketaux import (
    MarketauxClient,
)
from src.ingestion.news.upstox import (
    UpstoxNewsClient,
)
from src.ingestion.news.dedup import (
    deduplicate_news,
)

def check_asset(asset: str) -> None:
    marketaux = MarketauxClient()
    upstox = UpstoxNewsClient()

    marketaux_articles = marketaux.fetch_news(
        asset,
        limit=10,
    )

    upstox_articles = upstox.fetch_news(
        asset,
        page_size=10,
    )

    combined = (
        marketaux_articles
        + upstox_articles
    )

    result = deduplicate_news(combined)

    print(f"\n=== {asset} ===")
    print(
        "Marketaux:",
        len(marketaux_articles),
    )
    print(
        "Upstox:",
        len(upstox_articles),
    )
    print(
        "Combined:",
        len(combined),
    )
    print(
        "Unique:",
        len(result.articles),
    )
    print(
        "Duplicates:",
        result.duplicate_count,
    )

    print("\nUnique articles:")
    for article in result.articles:
        print(
            article.published_at,
            "|",
            article.source,
            "|",
            article.headline,
        )


def main():
    check_asset("TCS")
    check_asset("RELIANCE")


if __name__ == "__main__":
    main()