import pandas as pd

from src.features.sentiment_features import aggregate_sentiment


def test_sentiment_aggregation():
    df = pd.DataFrame(
        {
            "asset": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
            "exchange": [
                "NSE",
                "NSE",
                "NSE",
                "NSE",
                "NSE",
            ],
            "prediction_timestamp": pd.to_datetime(
                [
                    "2026-08-10 10:15:00",
                    "2026-08-10 10:15:00",
                    "2026-08-10 10:15:00",
                    "2026-08-10 10:15:00",
                    "2026-08-10 10:15:00",
                ]
            ),
            "sentiment_score": [
                0.75,
                -0.75,
                0.75,
                0.75,
                -0.75,
            ],
            "positive_probability": [
                0.80,
                0.05,
                0.80,
                0.80,
                0.05,
            ],
            "negative_probability": [
                0.05,
                0.80,
                0.05,
                0.05,
                0.80,
            ],
        }
    )

    result = aggregate_sentiment(df)

    aapl = result[
        (result["asset"] == "AAPL")
        & (result["prediction_timestamp"]
           == pd.Timestamp("2026-08-10 10:15:00"))
    ].iloc[0]

    msft = result[
        (result["asset"] == "MSFT")
        & (result["prediction_timestamp"]
           == pd.Timestamp("2026-08-10 10:15:00"))
    ].iloc[0]

    assert aapl["sentiment_mean"] == 0.25
    assert aapl["news_count"] == 3
    assert aapl["positive_ratio"] == 2 / 3
    assert aapl["negative_ratio"] == 1 / 3

    assert msft["sentiment_mean"] == 0.0
    assert msft["news_count"] == 2
    assert msft["positive_ratio"] == 0.5
    assert msft["negative_ratio"] == 0.5