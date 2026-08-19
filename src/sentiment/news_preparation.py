from dataclasses import dataclass
from datetime import datetime

from src.ingestion.schemas import NewsArticle


@dataclass(frozen=True)
class PreparedNewsArticle:
    asset: str
    exchange: str
    published_at: datetime
    source: str
    text: str


def prepare_news_for_sentiment(
    articles: list[NewsArticle],
) -> list[PreparedNewsArticle]:
    """
    Prepare unified news articles for sentiment inference.

    Uses article body text when available and falls
    back to the headline when the body is empty.

    Articles with neither usable body text nor
    headline text are discarded.
    """

    prepared: list[PreparedNewsArticle] = []

    for article in articles:
        body = (article.text or "").strip()
        headline = (
            article.headline or ""
        ).strip()

        if body:
            text = body
        elif headline:
            text = headline
        else:
            continue

        prepared.append(
            PreparedNewsArticle(
            asset=article.asset,
            exchange=article.exchange,
            published_at=article.published_at,
            source=article.source,
            text=text,)
        )

    return prepared