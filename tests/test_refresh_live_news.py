from datetime import datetime, timezone

import pandas as pd

from scripts.refresh_live_news import (
    CACHE_PATH,
    STATUS_PATH,
    _load_existing,
    refresh,
)
from src.ingestion.schemas import NewsArticle


def _article(asset, headline, url, published_at=None):
    ts = published_at or datetime.now(timezone.utc).isoformat()
    return NewsArticle(
        asset=asset,
        exchange="NSE",
        published_at=datetime.fromisoformat(ts),
        source="upstox",
        headline=headline,
        text="body",
        url=url,
    )


class FakeUpstox:
    def fetch_news(self, asset, page_size):
        return [_article(asset, f"{asset} headline", f"https://example.com/{asset.lower()}-1")]


class BrokenMarketaux:
    def fetch_news(self, asset, limit):
        raise RuntimeError("Marketaux down")


class FakeGdelt:
    def fetch_news(self, asset, max_records):
        return [_article(asset, f"{asset} headline", f"https://example.com/{asset.lower()}-1")]


def test_refresh_merges_and_writes_status(tmp_path, monkeypatch):
    import scripts.refresh_live_news as mod

    monkeypatch.setattr(mod, "CACHE_PATH", tmp_path / "live_news.csv")
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "live_news_status.json")

    class FakeClients:
        pass

    def fake_build():
        return (
            {"marketaux": BrokenMarketaux(), "upstox": FakeUpstox(), "gdelt": FakeGdelt()},
            {},
        )

    monkeypatch.setattr(mod, "_build_clients", fake_build)

    summary = refresh()

    assert summary["cache_rows"] == 5  # one deduped headline per asset
    assert summary["counts_by_asset"]["TCS"] == 1
    assert any("marketaux" in key for key in summary["provider_errors"])
    assert (tmp_path / "live_news.csv").exists()
    assert (tmp_path / "live_news_status.json").exists()


def test_refresh_retention_and_dedup(tmp_path, monkeypatch):
    import scripts.refresh_live_news as mod

    cache = tmp_path / "live_news.csv"
    monkeypatch.setattr(mod, "CACHE_PATH", cache)
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "live_news_status.json")

    old = pd.DataFrame(
        [
            {
                "asset": "TCS",
                "exchange": "NSE",
                "published_at": "2026-01-01T00:00:00+00:00",
                "source": "upstox",
                "headline": "ancient",
                "text": "",
                "url": "https://example.com/ancient",
            }
        ]
    )
    old.to_csv(cache, index=False)

    def fake_build():
        return ({"upstox": FakeUpstox()}, {})

    monkeypatch.setattr(mod, "_build_clients", fake_build)
    summary = refresh()

    df = pd.read_csv(cache)
    assert "ancient" not in df["headline"].tolist()  # retention pruned
    assert summary["cache_rows"] >= 5
