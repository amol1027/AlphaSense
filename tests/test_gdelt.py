from datetime import datetime, timezone
import requests

from src.ingestion.news.gdelt import (
    GDELTNewsClient,
)


def test_gdelt_fetch_news(monkeypatch):
    payload = {
        "articles": [
            {
                "url": (
                    "https://example.com/tcs"
                ),
                "title": (
                    "TCS announces new AI initiative"
                ),
                "seendate": (
                    "20260818T071000Z"
                ),
                "domain": "example.com",
                "language": "English",
                "sourcecountry": "India",
            }
        ]
    }

    class FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    client = GDELTNewsClient()

    articles = client.fetch_news(
        "TCS"
    )

    assert len(articles) == 1

    article = articles[0]

    assert article.asset == "TCS"
    assert article.exchange == "NSE"
    assert article.source == (
        "gdelt:example.com"
    )

    assert article.headline == (
        "TCS announces new AI initiative"
    )

    assert article.text == ""

    assert str(article.url) == (
        "https://example.com/tcs"
    )

    assert article.published_at.tzinfo == (
        timezone.utc
    )


def test_gdelt_rejects_unknown_asset():
    client = GDELTNewsClient()

    try:
        client.fetch_news("INFY")
        assert False
    except ValueError as exc:
        assert "Unsupported asset" in str(exc)


def test_gdelt_skips_incomplete_articles(
    monkeypatch,
):
    payload = {
        "articles": [
            {
                "title": "Missing URL",
                "seendate": (
                    "20260818T071000Z"
                ),
            },
            {
                "url": (
                    "https://example.com/valid"
                ),
                "title": "Valid article",
                "seendate": (
                    "20260818T072000Z"
                ),
                "domain": "example.com",
            },
        ]
    }

    class FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    client = GDELTNewsClient()

    articles = client.fetch_news(
        "TCS"
    )

    assert len(articles) == 1
    assert articles[0].headline == (
        "Valid article"
    )


def test_gdelt_empty_response(
    monkeypatch,
):
    class FakeResponse:
        status_code = 200
        def raise_for_status(self):
            pass

        def json(self):
            return {"articles": []}

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    client = GDELTNewsClient()

    articles = client.fetch_news(
        "TCS"
    )

    assert articles == []

def test_gdelt_retries_rate_limit(
    monkeypatch,
):
    responses = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code == 429:
                raise requests.HTTPError(
                    "429"
                )

        def json(self):
            return {
                "articles": [
                    {
                        "url": (
                            "https://example.com/tcs"
                        ),
                        "title": "TCS news",
                        "seendate": (
                            "20260818T071000Z"
                        ),
                        "domain": "example.com",
                    }
                ]
            }

    responses.extend(
        [
            FakeResponse(429),
            FakeResponse(200),
        ]
    )

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )

    monkeypatch.setattr(
        "src.ingestion.news.gdelt.time.sleep",
        lambda _: None,
    )

    client = GDELTNewsClient(
        max_retries=1,
        backoff_seconds=1,
    )

    articles = client.fetch_news("TCS")

    assert len(articles) == 1
    assert articles[0].headline == "TCS news"

def test_gdelt_historical_date_parameters(
    monkeypatch,
):
    captured = {}

    payload = {
        "articles": []
    }

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def fake_get(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )

    client = GDELTNewsClient()

    client.fetch_news(
        "TCS",
        max_records=25,
        start_datetime=datetime(
            2026,
            7,
            1,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        end_datetime=datetime(
            2026,
            8,
            10,
            23,
            59,
            59,
            tzinfo=timezone.utc,
        ),
    )

    params = captured["params"]

    assert params["maxrecords"] == 25

    assert params["startdatetime"] == (
        "20260701000000"
    )

    assert params["enddatetime"] == (
        "20260810235959"
    )