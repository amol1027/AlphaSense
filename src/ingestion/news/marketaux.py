import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from src.ingestion.schemas import NewsArticle


MARKETAUX_URL = (
    "https://api.marketaux.com/v1/news/all"
)

MARKETAUX_SYMBOLS = {
    "TCS": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
}


class MarketauxClient:
    def __init__(
        self,
        api_token: str | None = None,
        timeout: int = 30,
    ):
        load_dotenv()

        self.api_token = (
            api_token
            or os.getenv("MARKETAUX_API_TOKEN")
        )

        if not self.api_token:
            raise ValueError(
                "MARKETAUX_API_TOKEN is not set."
            )

        self.timeout = timeout

    def fetch_news(
        self,
        asset: str,
        limit: int = 50,
        published_after: str | None = None,
        published_before: str | None = None,
        page: int = 1,
    ) -> list[NewsArticle]:
        """
        Fetch news for a supported asset.

        Optional published_after and published_before
        parameters allow historical date filtering.
        Dates should be supplied in a format accepted
        by the Marketaux API.
        """

        if asset not in MARKETAUX_SYMBOLS:
            raise ValueError(
                f"Unsupported asset: {asset}"
            )

        symbol = MARKETAUX_SYMBOLS[asset]

        params = {
            "api_token": self.api_token,
            "symbols": symbol,
            "limit": limit,
            "language": "en",
            "page": page,
        }

        if published_after:
            params["published_after"] = (
                published_after
            )

        if published_before:
            params["published_before"] = (
                published_before
            )

        response = requests.get(
            MARKETAUX_URL,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        payload = response.json()

        articles = []

        for item in payload.get(
            "data",
            [],
        ):
            entities = item.get(
                "entities",
                [],
            )

            target_entity = any(
                entity.get("symbol") == symbol
                for entity in entities
            )

            if not target_entity:
                continue

            published_at = datetime.fromisoformat(
                item["published_at"].replace(
                    "Z",
                    "+00:00",
                )
            ).astimezone(
                timezone.utc
            )

            description = (
                item.get("description")
                or ""
            ).strip()

            snippet = (
                item.get("snippet")
                or ""
            ).strip()

            title = (
                item.get("title")
                or ""
            ).strip()

            text_parts = [
                part
                for part in [
                    description,
                    snippet,
                ]
                if part
            ]

            text = "\n\n".join(
                text_parts
            )

            articles.append(
                NewsArticle(
                    asset=asset,
                    exchange="NSE",
                    published_at=published_at,
                    source=(
                        f"marketaux:"
                        f"{item.get('source', '')}"
                    ),
                    headline=title,
                    text=text,
                    url=item["url"],
                )
            )

        return articles