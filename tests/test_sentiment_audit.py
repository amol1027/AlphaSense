import pandas as pd

from src.sentiment.audit import (
    audit_sentiment_data,
)


def make_dataframe():
    return pd.DataFrame(
        [
            {
                "asset": "TCS",
                "exchange": "NSE",
                "published_at": (
                    "2026-08-18T07:00:00Z"
                ),
                "source": "gdelt:test.com",
                "text": "TCS reports strong growth.",
                "positive_probability": 0.80,
                "neutral_probability": 0.15,
                "negative_probability": 0.05,
                "sentiment_score": 0.75,
            }
        ]
    )


def write_dataframe(
    tmp_path,
    dataframe,
):
    path = tmp_path / "sentiment.csv"
    dataframe.to_csv(
        path,
        index=False,
    )
    return str(path)


def test_valid_sentiment_data_passes(
    tmp_path,
):
    result = audit_sentiment_data(
        write_dataframe(
            tmp_path,
            make_dataframe(),
        )
    )

    assert result.passed
    assert result.rows == 1


def test_invalid_timestamp_is_detected(
    tmp_path,
):
    df = make_dataframe()
    df.loc[0, "published_at"] = (
        "not-a-timestamp"
    )

    result = audit_sentiment_data(
        write_dataframe(tmp_path, df)
    )

    assert result.invalid_timestamps == 1
    assert not result.passed


def test_invalid_asset_is_detected(
    tmp_path,
):
    df = make_dataframe()
    df.loc[0, "asset"] = "INFY"

    result = audit_sentiment_data(
        write_dataframe(tmp_path, df)
    )

    assert result.invalid_assets == 1
    assert not result.passed


def test_invalid_exchange_is_detected(
    tmp_path,
):
    df = make_dataframe()
    df.loc[0, "exchange"] = "BSE"

    result = audit_sentiment_data(
        write_dataframe(tmp_path, df)
    )

    assert result.invalid_exchanges == 1
    assert not result.passed


def test_probability_outside_range_is_detected(
    tmp_path,
):
    df = make_dataframe()
    df.loc[0, "positive_probability"] = 1.2

    result = audit_sentiment_data(
        write_dataframe(tmp_path, df)
    )

    assert result.invalid_probabilities == 1
    assert not result.passed


def test_probability_sum_violation_is_detected(
    tmp_path,
):
    df = make_dataframe()
    df.loc[0, "neutral_probability"] = 0.30

    result = audit_sentiment_data(
        write_dataframe(tmp_path, df)
    )

    assert result.probability_sum_violations == 1
    assert not result.passed


def test_score_violation_is_detected(
    tmp_path,
):
    df = make_dataframe()
    df.loc[0, "sentiment_score"] = 0.25

    result = audit_sentiment_data(
        write_dataframe(tmp_path, df)
    )

    assert result.score_violations == 1
    assert not result.passed


def test_duplicate_record_is_detected(
    tmp_path,
):
    df = pd.concat(
        [
            make_dataframe(),
            make_dataframe(),
        ],
        ignore_index=True,
    )

    result = audit_sentiment_data(
        write_dataframe(tmp_path, df)
    )

    assert result.duplicate_records == 1
    assert not result.passed


def test_empty_text_is_detected(
    tmp_path,
):
    df = make_dataframe()
    df.loc[0, "text"] = "   "

    result = audit_sentiment_data(
        write_dataframe(tmp_path, df)
    )

    assert result.empty_text == 1
    assert not result.passed