from datetime import datetime

from src.features.time_windows import filter_news_for_prediction
from src.ingestion.loader import load_news


def test_future_news_is_excluded():
    articles = load_news("data/raw/news_sample.csv")

    prediction_timestamp = datetime(
        2026,
        8,
        10,
        10,
        15,
    )

    eligible_articles = filter_news_for_prediction(
        articles,
        prediction_timestamp,
    )

    headlines = [
        article.headline
        for article in eligible_articles
    ]

    assert "TCS reports strong growth" in headlines
    assert "TCS wins major contract" in headlines

    # Published at 10:20 — after the 10:15 cutoff.
    assert "TCS revenue rises 20 percent" not in headlines

def test_news_becomes_available_to_later_prediction():
    articles = load_news("data/raw/news_sample.csv")

    prediction_timestamp = datetime(
        2026,
        8,
        10,
        10,
        30,
    )

    eligible_articles = filter_news_for_prediction(
        articles,
        prediction_timestamp,
    )

    headlines = [
        article.headline
        for article in eligible_articles
    ]

    assert "TCS revenue rises 20 percent" in headlines