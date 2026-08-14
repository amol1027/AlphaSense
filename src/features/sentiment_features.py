import pandas as pd


def aggregate_sentiment(
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate article-level sentiment by asset.

    Expected columns:
        asset
        published_at
        sentiment_score
        positive_probability
        negative_probability
    """

    if sentiment_df.empty:
        return pd.DataFrame(
            columns=[
                "asset",
                "sentiment_mean",
                "sentiment_std",
                "news_count",
                "positive_ratio",
                "negative_ratio",
            ]
        )

    aggregated = (
        sentiment_df
        .groupby("asset")
        .agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            news_count=("sentiment_score", "count"),
            positive_ratio=(
                "positive_probability",
                lambda x: (x >= 0.5).mean(),
            ),
            negative_ratio=(
                "negative_probability",
                lambda x: (x >= 0.5).mean(),
            ),
        )
        .reset_index()
    )

    # If an asset has only one article, standard deviation is NaN.
    aggregated["sentiment_std"] = (
        aggregated["sentiment_std"].fillna(0.0)
    )

    return aggregated