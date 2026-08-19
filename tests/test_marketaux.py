from datetime import timezone

import requests

from src.ingestion.news.marketaux import (
    MarketauxClient,
)


def test_marketaux_symbol_mapping():
    client = MarketauxClient(
        api_token="test-token"
    )

    assert client.api_token == "test-token"


def test_marketaux_rejects_unknown_asset():
    client = MarketauxClient(
        api_token="test-token"
    )

    try:
        client.fetch_news("INFY")
        assert False
    except ValueError as exc:
        assert "Unsupported asset" in str(exc)


def test_marketaux_fetch_news(monkeypatch):
    payload = {
        "data": [
            {
                "uuid": "test-uuid",
                "title": "TCS reports strong results",
                "description": "TCS revenue increased.",
                "snippet": "Strong quarterly performance.",
                "url": (
                    "https://example.com/tcs"
                ),
                "published_at": (
                    "2026-08-18T07:10:37.000000Z"
                ),
                "source": "example.com",
                "entities": [
                    {
                        "symbol": "TCS.NS",
                        "name": (
                            "Tata Consultancy "
                            "Services Limited"
                        ),
                    }
                ],
            }
        ]
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )

    client = MarketauxClient(
        api_token="test-token"
    )

    articles = client.fetch_news(
        "TCS"
    )

    assert len(articles) == 1

    article = articles[0]

    assert article.asset == "TCS"
    assert article.exchange == "NSE"
    assert article.headline == (
        "TCS reports strong results"
    )
    assert "TCS revenue increased." in (
        article.text
    )
    assert article.published_at.tzinfo == (
        timezone.utc
    )