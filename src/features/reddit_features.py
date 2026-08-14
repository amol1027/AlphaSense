import pandas as pd


def aggregate_reddit(
    reddit_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate Reddit post-level sentiment and engagement
    by asset, exchange, and prediction timestamp.

    Expected columns:
        asset
        exchange
        published_at
        prediction_timestamp
        sentiment_score
        positive_probability
        negative_probability
        score
        comments
    """

    output_columns = [
        "asset",
        "exchange",
        "prediction_timestamp",
        "reddit_sentiment_mean",
        "reddit_sentiment_std",
        "reddit_count",
        "reddit_positive_ratio",
        "reddit_negative_ratio",
        "reddit_score_mean",
        "reddit_comments_mean",
        "reddit_engagement_mean",
    ]

    if reddit_df.empty:
        return pd.DataFrame(columns=output_columns)

    df = reddit_df.copy()

    df["engagement"] = (
        df["score"] + df["comments"]
    )

    aggregated = (
        df.groupby(
            [
                "asset",
                "exchange",
                "prediction_timestamp",
            ]
        )
        .agg(
            reddit_sentiment_mean=(
                "sentiment_score",
                "mean",
            ),
            reddit_sentiment_std=(
                "sentiment_score",
                "std",
            ),
            reddit_count=(
                "sentiment_score",
                "count",
            ),
            reddit_positive_ratio=(
                "positive_probability",
                lambda x: (x >= 0.5).mean(),
            ),
            reddit_negative_ratio=(
                "negative_probability",
                lambda x: (x >= 0.5).mean(),
            ),
            reddit_score_mean=(
                "score",
                "mean",
            ),
            reddit_comments_mean=(
                "comments",
                "mean",
            ),
            reddit_engagement_mean=(
                "engagement",
                "mean",
            ),
        )
        .reset_index()
    )

    aggregated["reddit_sentiment_std"] = (
        aggregated["reddit_sentiment_std"]
        .fillna(0.0)
    )

    return aggregated