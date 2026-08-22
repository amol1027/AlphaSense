import pandas as pd

from src.features.market_features import (
    add_normalized_market_features,
)
from src.features.reddit_features import aggregate_reddit
from src.features.sentiment_features import aggregate_sentiment
from src.features.session_windows import (
    is_valid_prediction_timestamp,
)
from src.features.targets import add_next_hour_target
from src.features.time_windows import (
    filter_information_for_prediction,
)
from src.ingestion.loader import load_news
from src.ingestion.reddit_loader import load_reddit
from src.sentiment.finbert import (
    FinBERTSentimentProvider,
)
from src.sentiment.news_preparation import (
    prepare_news_for_sentiment,
)
from src.features.trading_calendar import (
    TradingCalendar,
    load_default_nse_bse_calendar,
)


def _normalize_utc_timestamp(
    value,
) -> pd.Timestamp:
    """
    Convert a timestamp to timezone-aware UTC.

    Used for information-cutoff comparisons.
    """
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def build_hourly_features(
    market_path: str,
    news_path: str,
    reddit_path: str,
    calendar: TradingCalendar | None = None,
    news_sentiment_path: str | None = None,
) -> pd.DataFrame:
    """
    Build market, normalized market, news sentiment,
    Reddit sentiment, and next-hour target features.

    Historical news sentiment can be supplied through
    news_sentiment_path. When supplied, precomputed
    FinBERT sentiment scores are used instead of
    running FinBERT again.

    News is eligible for a prediction when:

        news asset == market asset

    and:

        prediction_timestamp - 1 hour
        <= news published_at
        <= prediction_timestamp

    News exchange metadata is deliberately not used
    when determining eligibility.
    """

    calendar = (
        calendar
        if calendar is not None
        else load_default_nse_bse_calendar()
    )

    # ==================================================
    # 1. Load market data
    # ==================================================

    market_df = pd.read_csv(
        market_path
    )

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

    # ==================================================
    # 2. Add normalized market features
    # ==================================================

    market_df = add_normalized_market_features(
        market_df
    )

    # ==================================================
    # 3. Add next-hour target
    # ==================================================

    market_df = add_next_hour_target(
        market_df
    )

    # ==================================================
    # 4. Determine valid prediction timestamps
    # ==================================================

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

    # ==================================================
    # 5. Prepare news sentiment
    # ==================================================

    if news_sentiment_path is not None:

        sentiment_source = pd.read_csv(
            news_sentiment_path
        )

        required_columns = {
            "asset",
            "published_at",
            "sentiment_score",
            "positive_probability",
            "negative_probability",
        }

        missing = (
            required_columns
            - set(sentiment_source.columns)
        )

        if missing:
            raise ValueError(
                "Missing required sentiment columns: "
                f"{sorted(missing)}"
            )

        sentiment_source[
            "published_at"
        ] = pd.to_datetime(
            sentiment_source[
                "published_at"
            ],
            utc=True,
        )

        article_sentiments = (
            sentiment_source[
                [
                    "asset",
                    "published_at",
                    "sentiment_score",
                    "positive_probability",
                    "negative_probability",
                ]
            ]
            .to_dict("records")
        )

    else:

        articles = load_news(
            news_path
        )

        prepared_articles = (
            prepare_news_for_sentiment(
                articles
            )
        )

        article_sentiments = []

        if prepared_articles:

            sentiment_provider = (
                FinBERTSentimentProvider()
            )

            texts = [
                article.text
                for article in prepared_articles
            ]

            sentiments = (
                sentiment_provider.predict_batch(
                    texts
                )
            )

            article_sentiments = [
                {
                    "asset": article.asset,
                    "published_at": (
                        article.published_at
                    ),
                    "sentiment_score": (
                        sentiment.sentiment_score
                    ),
                    "positive_probability": (
                        sentiment.positive_probability
                    ),
                    "negative_probability": (
                        sentiment.negative_probability
                    ),
                }
                for article, sentiment in zip(
                    prepared_articles,
                    sentiments,
                )
            ]

    # ==================================================
    # 6. Associate news with predictions
    # ==================================================

    sentiment_rows = []

    for _, prediction in (
        prediction_timestamps.iterrows()
    ):

        prediction_timestamp = (
            prediction["timestamp"]
        )

        prediction_timestamp_utc = (
            _normalize_utc_timestamp(
                prediction_timestamp
            )
        )

        window_start_utc = (
            prediction_timestamp_utc
            - pd.Timedelta(hours=1)
        )

        for article in article_sentiments:

            if (
                article["asset"]
                != prediction["asset"]
            ):
                continue

            article_published_at = (
                _normalize_utc_timestamp(
                    article["published_at"]
                )
            )

            if (
                window_start_utc
                <= article_published_at
                <= prediction_timestamp_utc
            ):
                sentiment_rows.append(
                    {
                        "asset": article["asset"],
                        "exchange": (
                            prediction["exchange"]
                        ),
                        "prediction_timestamp": (
                            prediction_timestamp
                        ),
                        "sentiment_score": (
                            article["sentiment_score"]
                        ),
                        "positive_probability": (
                            article[
                                "positive_probability"
                            ]
                        ),
                        "negative_probability": (
                            article[
                                "negative_probability"
                            ]
                        ),
                    }
                )

    sentiment_df = pd.DataFrame(
        sentiment_rows
    )

    # ==================================================
    # 7. Aggregate news sentiment
    # ==================================================

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

        sentiment_features = (
            aggregate_sentiment(
                sentiment_df
            )
        )

    sentiment_features[
        "prediction_timestamp"
    ] = pd.to_datetime(
        sentiment_features[
            "prediction_timestamp"
        ]
    )

    # ==================================================
    # 8. Load Reddit
    # ==================================================

    reddit_posts = load_reddit(
        reddit_path
    )

    reddit_sentiment_provider = (
        FinBERTSentimentProvider()
    )

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

            result = (
                reddit_sentiment_provider.predict(
                    post.text
                )
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

    # ==================================================
    # 9. Aggregate Reddit features
    # ==================================================

    if reddit_df.empty:

        reddit_features = pd.DataFrame(
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

    else:

        reddit_features = aggregate_reddit(
            reddit_df
        )

    reddit_features[
        "prediction_timestamp"
    ] = pd.to_datetime(
        reddit_features[
            "prediction_timestamp"
        ]
    )

    # ==================================================
    # 10. Build base feature dataframe
    # ==================================================

    features = market_df.rename(
        columns={
            "timestamp": "prediction_timestamp"
        }
    ).copy()

    features[
        "prediction_timestamp"
    ] = pd.to_datetime(
        features[
            "prediction_timestamp"
        ]
    )

    # ==================================================
    # 11. Keep only valid prediction rows
    # ==================================================

    valid_predictions = (
        prediction_timestamps.rename(
            columns={
                "timestamp": (
                    "prediction_timestamp"
                )
            }
        )
        .copy()
    )

    valid_predictions[
        "prediction_timestamp"
    ] = pd.to_datetime(
        valid_predictions[
            "prediction_timestamp"
        ]
    )

    valid_keys = pd.MultiIndex.from_frame(
        valid_predictions[
            [
                "asset",
                "exchange",
                "prediction_timestamp",
            ]
        ]
    )

    feature_keys = pd.MultiIndex.from_frame(
        features[
            [
                "asset",
                "exchange",
                "prediction_timestamp",
            ]
        ]
    )

    features = features[
        feature_keys.isin(valid_keys)
    ].copy()

    features = features.reset_index(
        drop=True
    )

    # ==================================================
    # 12. Merge news features
    # ==================================================

    features = features.merge(
        sentiment_features,
        on=[
            "asset",
            "exchange",
            "prediction_timestamp",
        ],
        how="left",
    )

    # ==================================================
    # 13. Merge Reddit features
    # ==================================================

    features = features.merge(
        reddit_features,
        on=[
            "asset",
            "exchange",
            "prediction_timestamp",
        ],
        how="left",
    )

    # ==================================================
    # 14. Fill missing feature values
    # ==================================================

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