from datetime import datetime, timezone
import time

import requests

from src.ingestion.schemas import NewsArticle


GDELT_NEWS_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
)


GDELT_QUERIES = {
    "TCS": "Tata Consultancy Services",
    "RELIANCE": "Reliance Industries",
}


class GDELTNewsClient:
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_seconds: float = 5.0,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def fetch_news(
        self,
        asset: str,
        max_records: int = 10,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
    ) -> list[NewsArticle]:
        """
        Fetch news for a supported asset.

        Optional start_datetime and end_datetime
        restrict the GDELT search to a historical window.

        Datetimes are converted to UTC and formatted
        as GDELT YYYYMMDDHHMMSS timestamps.
        """

        if asset not in GDELT_QUERIES:
            raise ValueError(
                f"Unsupported asset: {asset}"
            )

        params = {
            "query": GDELT_QUERIES[asset],
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_records,
            "sort": "datedesc",
        }

        if start_datetime is not None:
            start_datetime = self._ensure_utc(
                start_datetime
            )

            params["startdatetime"] = (
                start_datetime.strftime(
                    "%Y%m%d%H%M%S"
                )
            )

        if end_datetime is not None:
            end_datetime = self._ensure_utc(
                end_datetime
            )

            params["enddatetime"] = (
                end_datetime.strftime(
                    "%Y%m%d%H%M%S"
                )
            )

        for attempt in range(
            self.max_retries + 1
        ):
            response = requests.get(
                GDELT_NEWS_URL,
                params=params,
                timeout=self.timeout,
            )

            if response.status_code != 429:
                response.raise_for_status()
                break

            if attempt == self.max_retries:
                response.raise_for_status()

            time.sleep(
                self.backoff_seconds
                * (2 ** attempt)
            )

        payload = response.json()

        articles = []

        for item in payload.get(
            "articles",
            [],
        ):
            url = item.get("url")
            title = item.get("title")
            seendate = item.get("seendate")

            if not url or not title or not seendate:
                continue

            published_at = datetime.strptime(
                seendate,
                "%Y%m%dT%H%M%SZ",
            ).replace(
                tzinfo=timezone.utc
            )

            domain = (
                item.get("domain")
                or "unknown"
            )

            articles.append(
                NewsArticle(
                    asset=asset,
                    exchange="NSE",
                    published_at=published_at,
                    source=f"gdelt:{domain}",
                    headline=title.strip(),
                    text="",
                    url=url,
                )
            )

        return articles

    @staticmethod
    def _ensure_utc(
        value: datetime,
    ) -> datetime:
        """
        Normalize a datetime to timezone-aware UTC.

        Naive datetimes are interpreted as UTC.
        """

        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )