from datetime import timezone

import requests

from src.ingestion.news.upstox import (
    UpstoxNewsClient,
)


def test_upstox_client_accepts_token():
    client = UpstoxNewsClient(
        access_token="test-token"
    )

    assert client.access_token == "test-token"


def test_upstox_rejects_unknown_asset():
    client = UpstoxNewsClient(
        access_token="test-token"
    )

    try:
        client.fetch_news("INFY")
        assert False
    except ValueError as exc:
        assert "Unsupported asset" in str(exc)


def test_upstox_fetch_news(monkeypatch):
    payload = {
        "status": "success",
        "data": {
            "NSE_EQ|INE467B01029": [
                {
                    "heading": (
                        "TCS launches AgentHub"
                    ),
                    "summary": (
                        "TCS announced a new AI platform."
                    ),
                    "article_link": (
                        "https://example.com/tcs"
                    ),
                    "published_time": (
                        1786948368547
                    ),
                }
            ]
        },
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

    client = UpstoxNewsClient(
        access_token="test-token"
    )

    articles = client.fetch_news(
        "TCS"
    )

    assert len(articles) == 1

    article = articles[0]

    assert article.asset == "TCS"
    assert article.exchange == "NSE"

    assert article.headline == (
        "TCS launches AgentHub"
    )

    assert article.text == (
        "TCS announced a new AI platform."
    )

    assert str(article.url) == (
    "https://example.com/tcs")

    assert article.published_at.tzinfo == (
        timezone.utc
    )


def test_upstox_skips_missing_timestamp(
    monkeypatch,
):
    payload = {
        "status": "success",
        "data": {
            "NSE_EQ|INE467B01029": [
                {
                    "heading": "Missing timestamp",
                    "summary": "Should be skipped",
                    "article_link": (
                        "https://example.com"
                    ),
                }
            ]
        },
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    client = UpstoxNewsClient(
        access_token="test-token"
    )

    articles = client.fetch_news("TCS")

    assert articles == []