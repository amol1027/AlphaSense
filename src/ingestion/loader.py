import pandas as pd

from .schemas import NewsArticle


def load_news(path: str) -> list[NewsArticle]:
    """Load and validate news articles from a CSV file."""

    df = pd.read_csv(path)

    required_columns = {
        "asset",
        "published_at",
        "source",
        "headline",
        "text",
        "url",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    articles = []

    for _, row in df.iterrows():
        article = NewsArticle(
            asset=row["asset"],
            published_at=row["published_at"],
            source=row["source"],
            headline=row["headline"],
            text=row["text"],
            url=row["url"],
        )

        articles.append(article)

    return articles