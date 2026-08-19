from dataclasses import dataclass
from datetime import datetime

from src.ingestion.schemas import NewsArticle
from src.ingestion.news.unified import UnifiedNewsClient
from src.sentiment.hourly_aggregation import (
    NewsSentimentRecord,
    HourlySentiment,
    aggregate_hourly_sentiment,
)
from src.sentiment.news_preparation import (
    prepare_news_for_sentiment,
)
from src.sentiment.finbert import (
    FinBERTSentimentProvider,
)


@dataclass(frozen=True)
class NewsSentimentPipelineResult:
    hourly_sentiment: list[HourlySentiment]
    usable_articles: int


class NewsSentimentPipeline:
    def __init__(
        self,
        sentiment_provider=None,
    ):
        self.sentiment_provider = (
            sentiment_provider
            or FinBERTSentimentProvider()
        )

    def process(
        self,
        articles: list[NewsArticle],
        prediction_timestamp: datetime,
    ) -> NewsSentimentPipelineResult:

        filtered_articles = (
            UnifiedNewsClient.filter_for_prediction(
                articles,
                prediction_timestamp,
            )
        )

        prepared = (
            prepare_news_for_sentiment(
                filtered_articles
            )
        )

        if not prepared:
            return NewsSentimentPipelineResult(
                hourly_sentiment=[],
                usable_articles=0,
            )

        texts = [
            article.text
            for article in prepared
        ]

        sentiments = (
            self.sentiment_provider.predict_batch(
                texts
            )
        )

        records = [
            NewsSentimentRecord(
                asset=article.asset,
                published_at=article.published_at,
                sentiment=sentiment,
            )
            for article, sentiment in zip(
                prepared,
                sentiments,
            )
        ]

        hourly = (
            aggregate_hourly_sentiment(
                records
            )
        )

        return NewsSentimentPipelineResult(
            hourly_sentiment=hourly,
            usable_articles=len(prepared),
        )