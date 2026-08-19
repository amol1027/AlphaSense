from datetime import datetime, timezone

from src.ingestion.news.dedup import (
    deduplicate_news,
    normalize_headline,
    normalize_url,
)
from src.ingestion.schemas import NewsArticle


def make_article(
    headline: str,
    url: str,
    asset: str = "TCS",
    published_at: str = (
        "2026-08-18T07:00:00+00:00"
    ),
):
    return NewsArticle(
        asset=asset,
        exchange="NSE",
        published_at=datetime.fromisoformat(
            published_at
        ),
        source="test",
        headline=headline,
        text="Test article",
        url=url,
    )


def test_normalize_url_removes_tracking_params():
    url = (
        "https://example.com/news/"
        "?utm_source=test&utm_medium=x&id=123"
    )

    result = normalize_url(url)

    assert result == (
        "https://example.com/news?id=123"
    )


def test_normalize_headline_collapses_whitespace():
    assert normalize_headline(
        "  TCS   launches   AgentHub "
    ) == "tcs launches agenthub"


def test_duplicate_urls_are_removed():
    articles = [
        make_article(
            "TCS news",
            "https://example.com/article",
        ),
        make_article(
            "Different headline",
            "https://example.com/article",
        ),
    ]

    result = deduplicate_news(articles)

    assert len(result.articles) == 1
    assert result.duplicate_count == 1


def test_tracking_url_variants_are_duplicates():
    articles = [
        make_article(
            "TCS news",
            "https://example.com/article",
        ),
        make_article(
            "Different headline",
            (
                "https://example.com/article"
                "?utm_source=marketaux"
            ),
        ),
    ]

    result = deduplicate_news(articles)

    assert len(result.articles) == 1
    assert result.duplicate_count == 1


def test_same_headline_near_in_time_is_duplicate():
    articles = [
        make_article(
            "TCS launches AgentHub",
            "https://marketaux.example/article",
        ),
        make_article(
            " TCS launches AgentHub ",
            "https://upstox.example/article",
            published_at=(
                "2026-08-18T07:05:00+00:00"
            ),
        ),
    ]

    result = deduplicate_news(articles)

    assert len(result.articles) == 1
    assert result.duplicate_count == 1


def test_same_headline_far_apart_is_not_duplicate():
    articles = [
        make_article(
            "TCS launches AgentHub",
            "https://example.com/1",
        ),
        make_article(
            "TCS launches AgentHub",
            "https://example.com/2",
            published_at=(
                "2026-08-18T08:00:00+00:00"
            ),
        ),
    ]

    result = deduplicate_news(articles)

    assert len(result.articles) == 2
    assert result.duplicate_count == 0


def test_different_assets_are_not_duplicate():
    articles = [
        make_article(
            "Markets fall",
            "https://example.com/tcs",
            asset="TCS",
        ),
        make_article(
            "Markets fall",
            "https://example.com/reliance",
            asset="RELIANCE",
        ),
    ]

    result = deduplicate_news(articles)

    assert len(result.articles) == 2
    assert result.duplicate_count == 0