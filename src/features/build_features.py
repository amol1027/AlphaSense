import pandas as pd

from src.features.sentiment_features import aggregate_sentiment
from src.features.session_windows import is_valid_prediction_timestamp
from src.features.targets import add_next_hour_target
from src.features.time_windows import filter_news_for_prediction
from src.ingestion.loader import load_news
from src.sentiment.dummy import DummySentimentProvider


def build_hourly_features(
    market_path: str,
    news_path: str,
) -> pd.DataFrame:
    """
    Build hourly market + news features and next-hour targets.

    The prediction timestamp is the market-data timestamp.
    Only news available at or before that timestamp is used.
    """

    market_df = pd.read_csv(market_path)
    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"])

    market_df = market_df.sort_values(["asset", "exchange", "timestamp"]).reset_index(
        drop=True
    )

    market_df = add_next_hour_target(market_df)
    articles = load_news(news_path)
    sentiment_provider = DummySentimentProvider()
    sentiment_records = []

    prediction_timestamps = market_df[["asset", "exchange", "timestamp"]].drop_duplicates()
    prediction_timestamps = prediction_timestamps[
        prediction_timestamps["timestamp"].apply(is_valid_prediction_timestamp)
    ].reset_index(drop=True)

    for _, prediction in prediction_timestamps.iterrows():
        prediction_timestamp = prediction["timestamp"]

        eligible_articles = filter_news_for_prediction(
            articles,
            prediction_timestamp.to_pydatetime(),
        )

        eligible_articles = [
            article
            for article in eligible_articles
            if (
                article.asset == prediction["asset"]
                and article.exchange == prediction["exchange"]
            )
        ]

        for article in eligible_articles:
            result = sentiment_provider.predict(article.text)

            sentiment_records.append(
                {
                    "asset": article.asset,
                    "exchange": article.exchange,
                    "published_at": article.published_at,
                    "prediction_timestamp": prediction_timestamp,
                    "positive_probability": result.positive_probability,
                    "negative_probability": result.negative_probability,
                    "sentiment_score": result.sentiment_score,
                }
            )

    sentiment_df = pd.DataFrame(sentiment_records)

    if sentiment_df.empty:
        sentiment_features = pd.DataFrame(
            columns=[
                "asset",
                "exchange",
                "prediction_timestamp",
                "sentiment_mean",
                "sentiment_std",
                "news_count",
                "positive_ratio",
                "negative_ratio",
            ]
        )
    else:
        sentiment_features = aggregate_sentiment(sentiment_df)

    valid_predictions = prediction_timestamps.rename(columns={"timestamp": "prediction_timestamp"})
    features = market_df.rename(columns={"timestamp": "prediction_timestamp"})

    features = features.merge(
        valid_predictions,
        on=["asset", "exchange", "prediction_timestamp"],
        how="inner",
    )

    features = features.merge(
        sentiment_features,
        on=["asset", "exchange", "prediction_timestamp"],
        how="left",
    )

    sentiment_columns = [
        "sentiment_mean",
        "sentiment_std",
        "news_count",
        "positive_ratio",
        "negative_ratio",
    ]

    for column in sentiment_columns:
        features[column] = features[column].fillna(0)

    return features