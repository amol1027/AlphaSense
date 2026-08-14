import pandas as pd

from src.ingestion.loader import load_news
from src.ingestion.market_loader import load_market_data


def test_load_news():
    articles = load_news("data/raw/news_sample.csv")

    assert len(articles) == 7

    first_article = articles[0]

    assert first_article.asset == "TCS"
    assert first_article.exchange == "NSE"
    assert first_article.source == "Example News"
    assert first_article.headline == "TCS reports strong growth"
    assert first_article.text != ""
    assert str(first_article.url) == "https://example.com/1"


def test_load_market_data():
    df = load_market_data(
        "data/raw/market_sample.csv"
    )

    assert len(df) == 14

    assert list(df.columns) == [
        "asset",
        "exchange",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert df.iloc[0]["timestamp"] == pd.Timestamp(
        "2026-08-10 09:15:00"
    )