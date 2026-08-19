from datetime import datetime, timezone

from src.ingestion.news.unified import (
    UnifiedNewsClient,
)
from src.ingestion.schemas import NewsArticle


def make_article(
    asset: str,
    source: str,
    headline: str,
    published_at: str,
    url: str,
):
    return NewsArticle(
        asset=asset,
        exchange="NSE",
        published_at=datetime.fromisoformat(
            published_at
        ),
        source=source,
        headline=headline,
        text="Test article",
        url=url,
    )


class FakeMarketaux:
    def fetch_news(self, asset, limit):
        return [
            make_article(
                asset,
                "marketaux:test.com",
                "Marketaux article",
                "2026-08-18T07:00:00+00:00",
                "https://example.com/a",
            )
        ]


class FakeUpstox:
    def fetch_news(
        self,
        asset,
        page_size,
    ):
        return [
            make_article(
                asset,
                "upstox",
                "Upstox article",
                "2026-08-18T07:05:00+00:00",
                "https://example.com/b",
            )
        ]


class FakeGDELT:
    def fetch_news(
        self,
        asset,
        max_records,
    ):
        return [
            make_article(
                asset,
                "gdelt:test.com",
                "GDELT article",
                "2026-08-18T07:10:00+00:00",
                "https://example.com/c",
            )
        ]


def test_unified_client_combines_providers():
    client = UnifiedNewsClient(
        marketaux_client=FakeMarketaux(),
        upstox_client=FakeUpstox(),
        gdelt_client=FakeGDELT(),
    )

    result = client.fetch_news("TCS")

    assert len(result.articles) == 3
    assert result.duplicate_count == 0
    assert result.provider_errors == {}


def test_unified_client_records_provider_failure():
    class BrokenMarketaux:
        def fetch_news(self, asset, limit):
            raise RuntimeError(
                "Marketaux unavailable"
            )

    client = UnifiedNewsClient(
        marketaux_client=BrokenMarketaux(),
        upstox_client=FakeUpstox(),
        gdelt_client=FakeGDELT(),
    )

    result = client.fetch_news("TCS")

    assert len(result.articles) == 2

    assert (
        "marketaux"
        in result.provider_errors
    )

    assert (
        "Marketaux unavailable"
        in result.provider_errors["marketaux"]
    )


def test_unified_client_deduplicates():
    class DuplicateUpstox:
        def fetch_news(
            self,
            asset,
            page_size,
        ):
            return [
                make_article(
                    asset,
                    "upstox",
                    "Same article",
                    "2026-08-18T07:00:00+00:00",
                    "https://example.com/a",
                )
            ]

    client = UnifiedNewsClient(
        marketaux_client=FakeMarketaux(),
        upstox_client=DuplicateUpstox(),
        gdelt_client=FakeGDELT(),
    )

    result = client.fetch_news("TCS")

    assert len(result.articles) == 2
    assert result.duplicate_count == 1


def test_filter_for_prediction_removes_future_news():
    articles = [
        make_article(
            "TCS",
            "test",
            "Past",
            "2026-08-18T06:00:00+00:00",
            "https://example.com/past",
        ),
        make_article(
            "TCS",
            "test",
            "At prediction time",
            "2026-08-18T07:00:00+00:00",
            "https://example.com/now",
        ),
        make_article(
            "TCS",
            "test",
            "Future",
            "2026-08-18T07:01:00+00:00",
            "https://example.com/future",
        ),
    ]

    prediction_time = (
        datetime(
            2026,
            8,
            18,
            7,
            0,
            tzinfo=timezone.utc,
        )
    )

    result = (
        UnifiedNewsClient.filter_for_prediction(
            articles,
            prediction_time,
        )
    )

    assert len(result) == 2

    assert [
        article.headline
        for article in result
    ] == [
        "Past",
        "At prediction time",
    ]