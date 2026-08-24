import pandas as pd

from src.ingestion.feature_audit import (
    audit_feature_data,
)


def make_features():
    return pd.DataFrame(
        {
            "asset": ["TCS", "TCS"],
            "exchange": ["NSE", "NSE"],
            "prediction_timestamp": pd.to_datetime(
                [
                    "2026-08-10 09:15:00+00:00",
                    "2026-08-10 10:15:00+00:00",
                ]
            ),
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1100],
            "return_15m": [0.01, 0.01],
            "return_30m": [0.02, 0.02],
            "return_1h": [0.03, 0.03],
            "high_low_range": [0.03, 0.03],
            "close_open_return": [0.01, 0.01],
            "volume_change": [0.1, 0.1],
            "future_timestamp": pd.to_datetime(
                [
                    "2026-08-10 10:15:00+00:00",
                    "2026-08-10 11:15:00+00:00",
                ]
            ),
            "target_return": [0.01, 0.02],
            "target_direction": [1, 1],
            "sentiment_mean": [0.2, 0.3],
            "sentiment_std": [0.1, 0.1],
            "news_count": [2, 3],
            "positive_ratio": [0.5, 0.7],
            "negative_ratio": [0.1, 0.2],
            "reddit_sentiment_mean": [0.1, 0.2],
            "reddit_sentiment_std": [0.1, 0.1],
            "reddit_count": [1, 2],
            "reddit_positive_ratio": [0.5, 0.5],
            "reddit_negative_ratio": [0.1, 0.1],
            "reddit_score_mean": [2, 3],
            "reddit_comments_mean": [1, 2],
            "reddit_engagement_mean": [3, 5],
        }
    )


def test_valid_feature_data_passes():
    result = audit_feature_data(
        make_features()
    )

    assert result.passed is True
    assert result.status == "PASS"


def test_duplicate_feature_keys_are_detected():
    df = pd.concat(
        [
            make_features(),
            make_features().iloc[[0]],
        ],
        ignore_index=True,
    )

    result = audit_feature_data(df)

    assert result.duplicate_keys == 1
    assert result.passed is False


def test_invalid_asset_is_detected():
    df = make_features()
    df.loc[0, "asset"] = "INFY"

    result = audit_feature_data(df)

    assert result.invalid_assets == 1
    assert result.passed is False


def test_invalid_exchange_is_detected():
    df = make_features()
    df.loc[0, "exchange"] = "BSE"

    result = audit_feature_data(df)

    assert result.invalid_exchanges == 1
    assert result.passed is False


def test_invalid_probability_ratio_is_detected():
    df = make_features()
    df.loc[0, "positive_ratio"] = 1.5

    result = audit_feature_data(df)

    assert result.invalid_sentiment_features == 1
    assert result.passed is False


def test_invalid_reddit_count_is_detected():
    df = make_features()
    df.loc[0, "reddit_count"] = -1

    result = audit_feature_data(df)

    assert result.invalid_reddit_features == 1
    assert result.passed is False


def test_missing_targets_are_reported():
    df = make_features()
    df.loc[1, "target_return"] = pd.NA
    df.loc[1, "target_direction"] = pd.NA

    result = audit_feature_data(df)

    assert result.missing_targets == 1
    assert result.passed is True


def test_invalid_target_direction_is_detected():
    df = make_features()
    df.loc[0, "target_direction"] = 2

    result = audit_feature_data(df)

    assert result.invalid_targets == 1
    assert result.passed is False


def test_invalid_target_alignment_is_detected():
    df = make_features()
    df.loc[0, "future_timestamp"] = pd.Timestamp(
        "2026-08-10 10:30:00+00:00"
    )

    result = audit_feature_data(df)

    assert result.invalid_target_alignment == 1
    assert result.passed is False