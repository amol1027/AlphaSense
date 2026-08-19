from datetime import datetime, timezone

from src.ingestion.schemas import NewsArticle
from src.sentiment.news_preparation import (
    prepare_news_for_sentiment,
)


def make_article(
    headline: str,
    text: str,
    asset: str = "TCS",
):
    return NewsArticle(
        asset=asset,
        exchange="NSE",
        published_at=datetime(
            2026,
            8,
            18,
            7,
            0,
            tzinfo=timezone.utc,
        ),
        source="test",
        headline=headline,
        text=text,
        url="https://example.com/news",
    )


def test_uses_article_body_when_available():
    articles = [
        make_article(
            headline="TCS headline",
            text="Full article body.",
        )
    ]

    result = prepare_news_for_sentiment(
        articles
    )

    assert len(result) == 1
    assert result[0].text == (
        "Full article body."
    )


def test_falls_back_to_headline():
    articles = [
        make_article(
            headline="TCS launches new AI platform",
            text="",
        )
    ]

    result = prepare_news_for_sentiment(
        articles
    )

    assert len(result) == 1
    assert result[0].text == (
        "TCS launches new AI platform"
    )


def test_skips_article_without_text():
    articles = [
        make_article(
            headline="",
            text="",
        )
    ]

    result = prepare_news_for_sentiment(
        articles
    )

    assert result == []


def test_preserves_metadata():
    articles = [
        make_article(
            headline="TCS news",
            text="Important details",
            asset="TCS",
        )
    ]

    result = prepare_news_for_sentiment(
        articles
    )

    assert result[0].asset == "TCS"
    assert result[0].source == "test"
    assert result[0].published_at == (
        datetime(
            2026,
            8,
            18,
            7,
            0,
            tzinfo=timezone.utc,
        )
    )