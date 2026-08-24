import pandas as pd
import pytest

from src.ingestion.news.audit import audit_news_data


def make_news_data():
    return pd.DataFrame(
        {
            "asset": ["TCS", "RELIANCE"],
            "exchange": ["NSE", "NSE"],
            "published_at": [
                "2026-08-10T07:00:00Z",
                "2026-08-10T08:00:00Z",
            ],
            "source": [
                "gdelt:example.com",
                "marketaux:example.com",
            ],
            "headline": [
                "TCS announces results",
                "Reliance launches project",
            ],
            "text": [
                "",
                "Some article text",
            ],
            "url": [
                "https://example.com/tcs",
                "https://example.com/reliance",
            ],
        }
    )


def test_valid_news_data_passes():
    result = audit_news_data(
        make_news_data(),
        start_date=pd.Timestamp("2026-08-01", tz="UTC"),
        end_date=pd.Timestamp("2026-08-11", tz="UTC"),
    )

    assert result.passed
    assert result.rows == 2
    assert result.invalid_timestamps == 0
    assert result.duplicate_urls == 0
    assert result.invalid_assets == 0


def test_missing_required_column_is_rejected():
    df = make_news_data().drop(columns=["headline"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        audit_news_data(df)


def test_invalid_timestamp_is_detected():
    df = make_news_data()
    df.loc[0, "published_at"] = "not-a-date"

    result = audit_news_data(df)

    assert result.invalid_timestamps == 1
    assert not result.passed


def test_duplicate_url_is_detected():
    df = make_news_data()
    df.loc[1, "url"] = df.loc[0, "url"]

    result = audit_news_data(df)

    assert result.duplicate_urls == 1
    assert not result.passed


def test_empty_headline_is_detected():
    df = make_news_data()
    df.loc[0, "headline"] = ""

    result = audit_news_data(df)

    assert result.empty_headlines == 1
    assert not result.passed


def test_invalid_asset_is_detected():
    df = make_news_data()
    df.loc[0, "asset"] = "INFY"

    result = audit_news_data(df)

    assert result.invalid_assets == 1
    assert not result.passed


def test_future_timestamp_is_detected():
    df = make_news_data()
    df.loc[0, "published_at"] = (
        "2027-01-01T00:00:00Z"
    )

    result = audit_news_data(
        df,
        end_date=pd.Timestamp(
            "2026-08-11",
            tz="UTC",
        ),
    )

    assert result.future_timestamps == 1
    assert not result.passed


def test_out_of_window_timestamp_is_detected():
    df = make_news_data()
    df.loc[0, "published_at"] = (
        "2026-07-01T00:00:00Z"
    )

    result = audit_news_data(
        df,
        start_date=pd.Timestamp(
            "2026-08-01",
            tz="UTC",
        ),
        end_date=pd.Timestamp(
            "2026-08-11",
            tz="UTC",
        ),
    )

    assert result.out_of_window_timestamps == 1
    assert not result.passed


def test_empty_body_is_reported_but_allowed():
    result = audit_news_data(
        make_news_data()
    )

    assert result.empty_bodies == 1
    assert result.passed