from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.sentiment.schemas import SentimentResult


@dataclass(frozen=True)
class NewsSentimentRecord:
    asset: str
    published_at: datetime
    sentiment: SentimentResult


@dataclass(frozen=True)
class HourlySentiment:
    asset: str
    hour: pd.Timestamp
    sentiment_mean: float
    sentiment_std: float
    news_count: int
    positive_ratio: float
    negative_ratio: float


def aggregate_hourly_sentiment(
    records: list[NewsSentimentRecord],
) -> list[HourlySentiment]:
    if not records:
        return []

    rows = []

    for record in records:
        timestamp = pd.Timestamp(
            record.published_at
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                "UTC"
            )
        else:
            timestamp = timestamp.tz_convert(
                "UTC"
            )

        rows.append(
            {
                "asset": record.asset,
                "published_at": timestamp,
                "sentiment_score": (
                    record.sentiment.sentiment_score
                ),
                "positive_probability": (
                    record.sentiment.positive_probability
                ),
                "negative_probability": (
                    record.sentiment.negative_probability
                ),
            }
        )

    frame = pd.DataFrame(rows)

    frame["hour"] = frame[
        "published_at"
    ].dt.floor("h")

    grouped = (
        frame
        .groupby(
            ["asset", "hour"],
            sort=True,
        )
        .agg(
            sentiment_mean=(
                "sentiment_score",
                "mean",
            ),
            sentiment_std=(
                "sentiment_score",
                "std",
            ),
            news_count=(
                "sentiment_score",
                "count",
            ),
            positive_ratio=(
                "positive_probability",
                lambda values: (
                    values >= 0.5
                ).mean(),
            ),
            negative_ratio=(
                "negative_probability",
                lambda values: (
                    values >= 0.5
                ).mean(),
            ),
        )
        .reset_index()
    )

    # A single article in an hour has zero
    # dispersion rather than NaN.
    grouped["sentiment_std"] = (
        grouped["sentiment_std"]
        .fillna(0.0)
    )

    return [
        HourlySentiment(
            asset=row.asset,
            hour=row.hour,
            sentiment_mean=float(
                row.sentiment_mean
            ),
            sentiment_std=float(
                row.sentiment_std
            ),
            news_count=int(
                row.news_count
            ),
            positive_ratio=float(
                row.positive_ratio
            ),
            negative_ratio=float(
                row.negative_ratio
            ),
        )
        for row in grouped.itertuples(
            index=False
        )
    ]