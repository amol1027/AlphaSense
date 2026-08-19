from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.features.time_windows import (
    filter_information_for_prediction,
)
from src.ingestion.news.dedup import (
    deduplicate_news,
)
from src.ingestion.news.gdelt import (
    GDELTNewsClient,
)
from src.ingestion.news.marketaux import (
    MarketauxClient,
)
from src.ingestion.news.upstox import (
    UpstoxNewsClient,
)
from src.ingestion.schemas import NewsArticle


@dataclass(frozen=True)
class NewsIngestionResult:
    articles: list[NewsArticle]
    provider_errors: dict[str, str]
    duplicate_count: int


class UnifiedNewsClient:
    def __init__(
        self,
        marketaux_client: MarketauxClient | None = None,
        upstox_client: UpstoxNewsClient | None = None,
        gdelt_client: GDELTNewsClient | None = None,
    ):
        self.marketaux = (
            marketaux_client
            or MarketauxClient()
        )

        self.upstox = (
            upstox_client
            or UpstoxNewsClient()
        )

        self.gdelt = (
            gdelt_client
            or GDELTNewsClient()
        )

    def fetch_news(
        self,
        asset: str,
        marketaux_limit: int = 10,
        upstox_page_size: int = 10,
        gdelt_max_records: int = 10,
    ) -> NewsIngestionResult:
        articles: list[NewsArticle] = []
        provider_errors: dict[str, str] = {}

        providers = [
            (
                "marketaux",
                lambda: self.marketaux.fetch_news(
                    asset,
                    limit=marketaux_limit,
                ),
            ),
            (
                "upstox",
                lambda: self.upstox.fetch_news(
                    asset,
                    page_size=upstox_page_size,
                ),
            ),
            (
                "gdelt",
                lambda: self.gdelt.fetch_news(
                    asset,
                    max_records=gdelt_max_records,
                ),
            ),
        ]

        for provider_name, fetcher in providers:
            try:
                articles.extend(fetcher())
            except Exception as exc:
                provider_errors[
                    provider_name
                ] = str(exc)

        result = deduplicate_news(
            articles
        )

        return NewsIngestionResult(
            articles=result.articles,
            provider_errors=provider_errors,
            duplicate_count=result.duplicate_count,
        )

    @staticmethod
    def filter_for_prediction(
    articles: list[NewsArticle],
    prediction_timestamp: pd.Timestamp | datetime,
) -> list[NewsArticle]:
        if isinstance(
            prediction_timestamp,
            pd.Timestamp,
    ):
            prediction_timestamp = (
            prediction_timestamp.to_pydatetime()
        )

        return filter_information_for_prediction(
        articles,
        prediction_timestamp,
    )