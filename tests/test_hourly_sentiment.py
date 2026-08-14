import pandas as pd

from src.ingestion.loader import load_news
from src.sentiment.dummy import DummySentimentProvider
from src.features.time_windows import filter_news_for_prediction
from src.features.sentiment_features import aggregate_sentiment


def test_tcs_1015_sentiment_features():
    articles = load_news(
        "data/raw/news_sample.csv"
    )

    prediction_timestamp = pd.Timestamp(
        "2026-08-10 10:15:00"
    )

    eligible_articles = filter_news_for_prediction(
        articles,
        prediction_timestamp.to_pydatetime(),
    )

    tcs_articles = [
        article
        for article in eligible_articles
        if article.asset == "TCS"
    ]

    provider = DummySentimentProvider()

    records = []

    for article in tcs_articles:
        result = provider.predict(article.text)

        records.append(
            {
                "asset": article.asset,
                "exchange": article.exchange,
                "published_at": article.published_at,
                "prediction_timestamp": prediction_timestamp,
                "positive_probability": (
                    result.positive_probability
                ),
                "negative_probability": (
                    result.negative_probability
                ),
                "sentiment_score": result.sentiment_score,
            }
        )

    sentiment_df = pd.DataFrame(records)

    result = aggregate_sentiment(
        sentiment_df
    )

    row = result.iloc[0]

    assert row["asset"] == "TCS"
    assert row["exchange"] == "NSE"
    assert row["news_count"] == 2

# The exact sentiment probabilities come from the dummy provider.
# We only verify that the ratios are valid.
    assert 0.0 <= row["positive_ratio"] <= 1.0
    assert 0.0 <= row["negative_ratio"] <= 1.0