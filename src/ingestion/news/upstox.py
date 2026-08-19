import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from src.ingestion.schemas import NewsArticle


UPSTOX_NEWS_URL = (
    "https://api.upstox.com/v2/news"
)

UPSTOX_INSTRUMENTS = {
    "TCS": "NSE_EQ|INE467B01029",
    "RELIANCE": "NSE_EQ|INE002A01018",
}


class UpstoxNewsClient:
    def __init__(
        self,
        access_token: str | None = None,
        timeout: int = 30,
    ):
        load_dotenv()

        self.access_token = (
            access_token
            or os.getenv("UPSTOX_ACCESS_TOKEN")
        )

        if not self.access_token:
            raise ValueError(
                "UPSTOX_ACCESS_TOKEN is not set."
            )

        self.timeout = timeout

    def fetch_news(
        self,
        asset: str,
        page_size: int = 10,
    ) -> list[NewsArticle]:
        if asset not in UPSTOX_INSTRUMENTS:
            raise ValueError(
                f"Unsupported asset: {asset}"
            )

        instrument_key = UPSTOX_INSTRUMENTS[
            asset
        ]

        response = requests.get(
            UPSTOX_NEWS_URL,
            headers={
                "Accept": "application/json",
                "Authorization": (
                    f"Bearer {self.access_token}"
                ),
            },
            params={
                "category": "instrument_keys",
                "instrument_keys": instrument_key,
                "page_number": 1,
                "page_size": page_size,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        payload = response.json()

        articles = []

        data = payload.get("data", {})
        items = data.get(instrument_key, [])

        for item in items:
            published_ms = item.get(
                "published_time"
            )

            if published_ms is None:
                continue

            published_at = datetime.fromtimestamp(
                published_ms / 1000,
                tz=timezone.utc,
            )

            headline = (
                item.get("heading") or ""
            ).strip()

            summary = (
                item.get("summary") or ""
            ).strip()

            articles.append(
                NewsArticle(
                    asset=asset,
                    exchange="NSE",
                    published_at=published_at,
                    source="upstox",
                    headline=headline,
                    text=summary,
                    url=item["article_link"],
                )
            )

        return articles