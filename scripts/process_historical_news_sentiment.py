from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.schemas import NewsArticle
from src.sentiment.finbert import FinBERTSentimentProvider
from src.sentiment.news_preparation import (
    prepare_news_for_sentiment,
)


INPUT_PATH = (
    "data/raw/news/research_news_expanded.csv"
)

OUTPUT_PATH = (
    "data/processed/research_news_sentiment.csv"
)


def load_articles(path: str) -> list[NewsArticle]:
    df = pd.read_csv(path)

    required_columns = {
        "asset",
        "exchange",
        "published_at",
        "source",
        "headline",
        "text",
        "url",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    articles = []

    for _, row in df.iterrows():
        published_at = pd.to_datetime(
            row["published_at"],
            utc=True,
        )

        text = (
            ""
            if pd.isna(row["text"])
            else str(row["text"])
        )

        headline = (
            ""
            if pd.isna(row["headline"])
            else str(row["headline"])
        )

        articles.append(
            NewsArticle(
                asset=str(row["asset"]),
                exchange=str(row["exchange"]),
                published_at=published_at,
                source=str(row["source"]),
                headline=headline,
                text=text,
                url=str(row["url"]),
            )
        )

    return articles


def main():
    print("Loading historical news...")

    articles = load_articles(INPUT_PATH)

    print(
        f"Loaded articles: {len(articles)}"
    )

    prepared = prepare_news_for_sentiment(
        articles
    )

    print(
        f"Usable articles: {len(prepared)}"
    )

    if not prepared:
        raise ValueError(
            "No usable articles found."
        )

    print(
        "Initializing FinBERT..."
    )

    provider = FinBERTSentimentProvider()

    texts = [
        article.text
        for article in prepared
    ]

    print(
        "Running FinBERT sentiment..."
    )

    sentiments = provider.predict_batch(
        texts
    )

    if len(sentiments) != len(prepared):
        raise RuntimeError(
            "Sentiment result count does not "
            "match prepared article count."
        )

    rows = []

    for article, sentiment in zip(
        prepared,
        sentiments,
    ):
        rows.append(
            {
                "asset": article.asset,
                "exchange": article.exchange,
                "published_at": (
                    pd.Timestamp(
                        article.published_at
                    )
                    .tz_convert("UTC")
                ),
                "source": article.source,
                "text": article.text,
                "positive_probability": (
                    sentiment.positive_probability
                ),
                "neutral_probability": (
                    sentiment.neutral_probability
                ),
                "negative_probability": (
                    sentiment.negative_probability
                ),
                "sentiment_score": (
                    sentiment.sentiment_score
                ),
            }
        )

    result = pd.DataFrame(rows)

    result["published_at"] = pd.to_datetime(
        result["published_at"],
        utc=True,
    )

    result = result.sort_values(
        [
            "asset",
            "published_at",
        ]
    ).reset_index(drop=True)

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "Saved:",
        output_path,
    )

    print()
    print(
        "Articles by asset:"
    )

    print(
        result["asset"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Sentiment summary:"
    )

    print(
        result[
            [
                "sentiment_score",
                "positive_probability",
                "neutral_probability",
                "negative_probability",
            ]
        ]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()