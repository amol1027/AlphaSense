from datetime import datetime, timezone

from src.ingestion.schemas import NewsArticle
from src.sentiment.pipeline import (
    NewsSentimentPipeline,
)
from src.sentiment.schemas import SentimentResult


def make_article(
    headline,
    text,
    published_at,
    asset="TCS",
):
    return NewsArticle(
        asset=asset,
        exchange="NSE",
        published_at=published_at,
        source="test",
        headline=headline,
        text=text,
        url=(
            "https://example.com/"
            + headline.replace(" ", "-")
        ),
    )


class FakeSentimentProvider:
    def predict_batch(self, texts):
        results = []

        for text in texts:
            if "strong" in text.lower():
                results.append(
                    SentimentResult(
                        positive_probability=0.9,
                        neutral_probability=0.08,
                        negative_probability=0.02,
                        sentiment_score=0.88,
                    )
                )
            else:
                results.append(
                    SentimentResult(
                        positive_probability=0.05,
                        neutral_probability=0.10,
                        negative_probability=0.85,
                        sentiment_score=-0.80,
                    )
                )

        return results


def test_pipeline_excludes_future_news():
    articles = [
        make_article(
            "Past article",
            "TCS reports strong growth.",
            datetime(
                2026,
                8,
                18,
                6,
                30,
                tzinfo=timezone.utc,
            ),
        ),
        make_article(
            "Future article",
            "TCS reports strong growth.",
            datetime(
                2026,
                8,
                18,
                7,
                30,
                tzinfo=timezone.utc,
            ),
        ),
    ]

    pipeline = NewsSentimentPipeline(
        sentiment_provider=(
            FakeSentimentProvider()
        )
    )

    result = pipeline.process(
        articles,
        datetime(
            2026,
            8,
            18,
            7,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert result.usable_articles == 1
    assert len(result.hourly_sentiment) == 1

    row = result.hourly_sentiment[0]

    assert row.asset == "TCS"
    assert row.news_count == 1
    assert row.sentiment_mean == 0.88


def test_pipeline_uses_headline_when_body_missing():
    articles = [
        make_article(
            "Strong TCS growth",
            "",
            datetime(
                2026,
                8,
                18,
                7,
                0,
                tzinfo=timezone.utc,
            ),
        )
    ]

    pipeline = NewsSentimentPipeline(
        sentiment_provider=(
            FakeSentimentProvider()
        )
    )

    result = pipeline.process(
        articles,
        datetime(
            2026,
            8,
            18,
            8,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert result.usable_articles == 1
    assert (
        result.hourly_sentiment[0]
        .sentiment_mean
        == 0.88
    )


def test_pipeline_returns_empty_for_no_usable_news():
    articles = [
        make_article(
            "",
            "",
            datetime(
                2026,
                8,
                18,
                7,
                0,
                tzinfo=timezone.utc,
            ),
        )
    ]

    pipeline = NewsSentimentPipeline(
        sentiment_provider=(
            FakeSentimentProvider()
        )
    )

    result = pipeline.process(
        articles,
        datetime(
            2026,
            8,
            18,
            8,
            0,
            tzinfo=timezone.utc,
        ),
    )

    assert result.usable_articles == 0
    assert result.hourly_sentiment == []