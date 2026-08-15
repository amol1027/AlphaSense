import pandas as pd

from src.features.reddit_features import aggregate_reddit
from src.features.sentiment_features import aggregate_sentiment
from src.features.session_windows import is_valid_prediction_timestamp
from src.features.targets import add_next_hour_target
from src.features.time_windows import (
    filter_information_for_prediction,
)
from src.ingestion.loader import load_news
from src.ingestion.reddit_loader import load_reddit
from src.sentiment.dummy import DummySentimentProvider
from src.features.trading_calendar import (
    TradingCalendar,
    WeekdayTradingCalendar,
)


def build_hourly_features(
    market_path: str,
    news_path: str,
    reddit_path: str,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """
    Build hourly market + news + Reddit features
    and next-hour targets.

    The prediction timestamp is the market-data timestamp.

    Only information available at or before the
    prediction timestamp is used.
    """
    calendar = (
    calendar
    if calendar is not None
    else WeekdayTradingCalendar() )

    # --------------------------------------------------
    # 1. Load market data
    # --------------------------------------------------

    market_df = pd.read_csv(market_path)

    market_df["timestamp"] = pd.to_datetime(
        market_df["timestamp"]
    )

    market_df = market_df.sort_values(
        [
            "asset",
            "exchange",
            "timestamp",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 2. Add next-hour targets
    # --------------------------------------------------

    market_df = add_next_hour_target(
        market_df
    )

    # --------------------------------------------------
    # 3. Determine valid prediction timestamps
    # --------------------------------------------------

    prediction_timestamps = (
        market_df[
            [
                "asset",
                "exchange",
                "timestamp",
            ]
        ]
        .drop_duplicates()
    )

    prediction_timestamps = (
        prediction_timestamps[
            prediction_timestamps["timestamp"].apply(
                lambda timestamp: (
                    is_valid_prediction_timestamp(
                        timestamp,
                        calendar,
                    )
                )
            )
        ]
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 4. Load news and Reddit
    # --------------------------------------------------

    articles = load_news(news_path)

    reddit_posts = load_reddit(
        reddit_path
    )

    sentiment_provider = (
        DummySentimentProvider()
    )

    # --------------------------------------------------
    # 5. Build news sentiment records
    # --------------------------------------------------

    sentiment_records = []

    for _, prediction in (
        prediction_timestamps.iterrows()
    ):

        prediction_timestamp = (
            prediction["timestamp"]
        )

        eligible_articles = (
            filter_information_for_prediction(
                articles,
                prediction_timestamp.to_pydatetime(),
            )
        )

        eligible_articles = [
            article
            for article in eligible_articles
            if (
                article.asset
                == prediction["asset"]
                and article.exchange
                == prediction["exchange"]
            )
        ]

        for article in eligible_articles:

            result = sentiment_provider.predict(
                article.text
            )

            sentiment_records.append(
                {
                    "asset": article.asset,
                    "exchange": article.exchange,
                    "published_at": (
                        article.published_at
                    ),
                    "prediction_timestamp": (
                        prediction_timestamp
                    ),
                    "positive_probability": (
                        result.positive_probability
                    ),
                    "negative_probability": (
                        result.negative_probability
                    ),
                    "sentiment_score": (
                        result.sentiment_score
                    ),
                }
            )

    sentiment_df = pd.DataFrame(
        sentiment_records
    )

    if sentiment_df.empty:

        sentiment_features = (
            pd.DataFrame(
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
        )
        sentiment_features["prediction_timestamp"] = pd.to_datetime(
        sentiment_features["prediction_timestamp"]
    )

    else:

        sentiment_features = (
            aggregate_sentiment(
                sentiment_df
            )
        )

    # --------------------------------------------------
    # 6. Build Reddit sentiment records
    # --------------------------------------------------

    reddit_records = []

    for _, prediction in (
        prediction_timestamps.iterrows()
    ):

        prediction_timestamp = (
            prediction["timestamp"]
        )

        eligible_posts = (
            filter_information_for_prediction(
                reddit_posts,
                prediction_timestamp.to_pydatetime(),
            )
        )

        eligible_posts = [
            post
            for post in eligible_posts
            if (
                post.asset
                == prediction["asset"]
                and post.exchange
                == prediction["exchange"]
            )
        ]

        for post in eligible_posts:

            result = sentiment_provider.predict(
                post.text
            )

            reddit_records.append(
                {
                    "asset": post.asset,
                    "exchange": post.exchange,
                    "published_at": (
                        post.published_at
                    ),
                    "prediction_timestamp": (
                        prediction_timestamp
                    ),
                    "positive_probability": (
                        result.positive_probability
                    ),
                    "negative_probability": (
                        result.negative_probability
                    ),
                    "sentiment_score": (
                        result.sentiment_score
                    ),
                    "score": post.score,
                    "comments": post.comments,
                }
            )

    reddit_df = pd.DataFrame(
        reddit_records
    )

    if reddit_df.empty:

        reddit_features = (
            pd.DataFrame(
                columns=[
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
            )
        )
        reddit_features["prediction_timestamp"] = pd.to_datetime(reddit_features["prediction_timestamp"])

    else:

        reddit_features = (
            aggregate_reddit(
                reddit_df
            )
        )

    # --------------------------------------------------
    # 7. Create valid market prediction rows
    # --------------------------------------------------

    valid_predictions = (
        prediction_timestamps.rename(
            columns={
                "timestamp":
                    "prediction_timestamp"
            }
        )
    )

    features = market_df.rename(
        columns={
            "timestamp":
                "prediction_timestamp"
        }
    )

    features = features.merge(
        valid_predictions,
        on=[
            "asset",
            "exchange",
            "prediction_timestamp",
        ],
        how="inner",
    )

    # --------------------------------------------------
    # 8. Merge news features
    # --------------------------------------------------

    features = features.merge(
        sentiment_features,
        on=[
            "asset",
            "exchange",
            "prediction_timestamp",
        ],
        how="left",
    )

    # --------------------------------------------------
    # 9. Merge Reddit features
    # --------------------------------------------------

    features = features.merge(
        reddit_features,
        on=[
            "asset",
            "exchange",
            "prediction_timestamp",
        ],
        how="left",
    )

    # --------------------------------------------------
    # 10. Fill missing social/news features
    # --------------------------------------------------

    feature_columns = [
        "sentiment_mean",
        "sentiment_std",
        "news_count",
        "positive_ratio",
        "negative_ratio",
        "reddit_sentiment_mean",
        "reddit_sentiment_std",
        "reddit_count",
        "reddit_positive_ratio",
        "reddit_negative_ratio",
        "reddit_score_mean",
        "reddit_comments_mean",
        "reddit_engagement_mean",
    ]

    for column in feature_columns:

        features[column] = (
            features[column]
            .fillna(0)
        )

    return features