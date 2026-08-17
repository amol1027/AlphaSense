import pandas as pd

from .schemas import NewsArticle



def load_news(path: str) -> list[NewsArticle]:
    """Load and validate news articles from a CSV file."""

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

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    articles = []

    

    for _, row in df.iterrows():
        published_at = pd.to_datetime(
        row["published_at"],
        utc=True,)
        article = NewsArticle(
            asset=row["asset"],
            exchange=row["exchange"],
            published_at=published_at,
            source=row["source"],
            headline=row["headline"],
            text=row["text"],
            url=row["url"],
        )

        articles.append(article)

    return articles