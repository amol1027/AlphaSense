import pandas as pd

from .reddit_schema import RedditPost


REQUIRED_COLUMNS = {
    "asset",
    "exchange",
    "published_at",
    "source",
    "headline",
    "text",
    "url",
    "score",
    "comments",
}


def load_reddit(path: str) -> list[RedditPost]:
    """Load and validate Reddit posts from CSV."""

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    posts = []

    for _, row in df.iterrows():
        published_at = pd.to_datetime(
        row["published_at"],
        utc=True, )
        post = RedditPost(
            asset=row["asset"],
            exchange=row["exchange"],
            published_at=published_at,
            source=row["source"],
            headline=row["headline"],
            text=row["text"],
            url=row["url"],
            score=int(row["score"]),
            comments=int(row["comments"]),
        )

        posts.append(post)

    return posts