import pandas as pd

from src.features.reddit_features import aggregate_reddit


def test_reddit_aggregation():
    df = pd.DataFrame(
        {
            "asset": [
                "TCS",
                "TCS",
                "TCS",
            ],
            "exchange": [
                "NSE",
                "NSE",
                "NSE",
            ],
            "published_at": pd.to_datetime(
                [
                    "2026-08-10 09:40:00",
                    "2026-08-10 10:05:00",
                    "2026-08-10 10:20:00",
                ]
            ),
            "prediction_timestamp": pd.to_datetime(
                [
                    "2026-08-10 10:15:00",
                    "2026-08-10 10:15:00",
                    "2026-08-10 10:15:00",
                ]
            ),
            "sentiment_score": [
                0.75,
                -0.25,
                0.50,
            ],
            "positive_probability": [
                0.80,
                0.20,
                0.60,
            ],
            "negative_probability": [
                0.05,
                0.70,
                0.10,
            ],
            "score": [
                25,
                40,
                80,
            ],
            "comments": [
                6,
                10,
                25,
            ],
        }
    )

    # Only the first two records should be considered
    # part of the 10:15 prediction.
    df = df[
        df["published_at"]
        <= pd.Timestamp("2026-08-10 10:15:00")
    ]

    result = aggregate_reddit(df)

    row = result.iloc[0]

    assert row["asset"] == "TCS"
    assert row["exchange"] == "NSE"

    assert row["reddit_count"] == 2

    expected_mean = (0.75 + (-0.25)) / 2

    assert row["reddit_sentiment_mean"] == expected_mean

    assert row["reddit_positive_ratio"] == 0.5

    assert row["reddit_negative_ratio"] == 0.5

    assert row["reddit_score_mean"] == 32.5

    assert row["reddit_comments_mean"] == 8.0

    assert row["reddit_engagement_mean"] == 40.5

def test_empty_reddit_aggregation():
    df = pd.DataFrame(
        columns=[
            "asset",
            "exchange",
            "published_at",
            "prediction_timestamp",
            "sentiment_score",
            "positive_probability",
            "negative_probability",
            "score",
            "comments",
        ]
    )

    result = aggregate_reddit(df)

    assert result.empty

    assert list(result.columns) == [
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