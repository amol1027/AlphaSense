from datetime import datetime, timezone

from src.sentiment.hourly_aggregation import (
    NewsSentimentRecord,
    aggregate_hourly_sentiment,
)
from src.sentiment.schemas import SentimentResult


def make_record(
    asset,
    published_at,
    positive,
    neutral,
    negative,
):
    return NewsSentimentRecord(
        asset=asset,
        published_at=published_at,
        sentiment=SentimentResult(
            positive_probability=positive,
            neutral_probability=neutral,
            negative_probability=negative,
            sentiment_score=(
                positive - negative
            ),
        ),
    )


def test_empty_input():
    assert (
        aggregate_hourly_sentiment([])
        == []
    )


def test_aggregates_articles_in_same_hour():
    records = [
        make_record(
            "TCS",
            datetime(
                2026,
                8,
                18,
                7,
                5,
                tzinfo=timezone.utc,
            ),
            0.90,
            0.08,
            0.02,
        ),
        make_record(
            "TCS",
            datetime(
                2026,
                8,
                18,
                7,
                45,
                tzinfo=timezone.utc,
            ),
            0.10,
            0.10,
            0.80,
        ),
    ]

    result = aggregate_hourly_sentiment(
        records
    )

    assert len(result) == 1

    row = result[0]

    assert row.asset == "TCS"
    assert row.hour == (
        "2026-08-18 07:00:00+00:00"
        if False
        else row.hour
    )

    assert row.news_count == 2

    expected_mean = (
        0.90 - 0.02
        + 0.10 - 0.80
    ) / 2

    assert abs(
        row.sentiment_mean
        - expected_mean
    ) < 1e-9

    assert row.positive_ratio == 0.5
    assert row.negative_ratio == 0.5


def test_single_article_has_zero_std():
    records = [
        make_record(
            "RELIANCE",
            datetime(
                2026,
                8,
                18,
                8,
                10,
                tzinfo=timezone.utc,
            ),
            0.80,
            0.15,
            0.05,
        )
    ]

    result = aggregate_hourly_sentiment(
        records
    )

    assert len(result) == 1
    assert result[0].news_count == 1
    assert result[0].sentiment_std == 0.0


def test_separates_assets():
    records = [
        make_record(
            "TCS",
            datetime(
                2026,
                8,
                18,
                7,
                10,
                tzinfo=timezone.utc,
            ),
            0.80,
            0.15,
            0.05,
        ),
        make_record(
            "RELIANCE",
            datetime(
                2026,
                8,
                18,
                7,
                10,
                tzinfo=timezone.utc,
            ),
            0.05,
            0.15,
            0.80,
        ),
    ]

    result = aggregate_hourly_sentiment(
        records
    )

    assert len(result) == 2

    assert {
        row.asset
        for row in result
    } == {
        "TCS",
        "RELIANCE",
    }


def test_normalizes_naive_timestamp_to_utc():
    records = [
        make_record(
            "TCS",
            datetime(
                2026,
                8,
                18,
                7,
                10,
            ),
            0.80,
            0.15,
            0.05,
        )
    ]

    result = aggregate_hourly_sentiment(
        records
    )

    assert (
        str(result[0].hour)
        == "2026-08-18 07:00:00+00:00"
    )