import pandas as pd

from src.modeling.baseline_runner import (
    BaselineEvaluation,
    evaluate_majority_baseline,
)
from src.modeling.dataset import (
    DEFAULT_FEATURE_COLUMNS,
)


def make_feature_data():
    rows = []

    timestamps = [
        "2026-08-01 10:15",
        "2026-08-05 10:15",
        "2026-08-12 10:15",
        "2026-08-13 10:15",
        "2026-08-20 10:15",
        "2026-08-21 10:15",
    ]

    targets = [
        1,
        1,
        0,
        0,
        1,
        0,
    ]

    for i, (timestamp, target) in enumerate(
        zip(timestamps, targets)
    ):
        rows.append(
            {
                "asset": "TCS",
                "exchange": "NSE",
                "prediction_timestamp": timestamp,
                "open": 100 + i,
                "high": 101 + i,
                "low": 99 + i,
                "close": 100.5 + i,
                "volume": 1000 + i,
                "future_close": 101 + i,
                "target_return": 0.01,
                "target_direction": target,
                "sentiment_mean": 0.1,
                "sentiment_std": 0.2,
                "news_count": 1,
                "positive_ratio": 1.0,
                "negative_ratio": 0.0,
                "reddit_sentiment_mean": 0.1,
                "reddit_sentiment_std": 0.2,
                "reddit_count": 1,
                "reddit_positive_ratio": 1.0,
                "reddit_negative_ratio": 0.0,
                "reddit_score_mean": 10.0,
                "reddit_comments_mean": 2.0,
                "reddit_engagement_mean": 12.0,
            }
        )

    return pd.DataFrame(rows)


def test_evaluate_majority_baseline():
    df = make_feature_data()

    result = evaluate_majority_baseline(
        df,
        train_end=pd.Timestamp(
            "2026-08-10"
        ),
        validation_end=pd.Timestamp(
            "2026-08-15"
        ),
    )

    assert isinstance(
        result,
        BaselineEvaluation,
    )

    # Training contains:
    # Aug 1 -> 1
    # Aug 5 -> 1
    #
    # Therefore majority class = 1.
    assert result.model.majority_class == 1


def test_baseline_uses_training_only():
    df = make_feature_data()

    result = evaluate_majority_baseline(
        df,
        train_end=pd.Timestamp(
            "2026-08-10"
        ),
        validation_end=pd.Timestamp(
            "2026-08-15"
        ),
    )

    # Validation labels are [0, 0].
    # Test labels are [1, 0].
    #
    # Neither can influence the learned
    # majority class.
    assert result.model.majority_class == 1


def test_validation_metrics_are_computed():
    df = make_feature_data()

    result = evaluate_majority_baseline(
        df,
        train_end=pd.Timestamp(
            "2026-08-10"
        ),
        validation_end=pd.Timestamp(
            "2026-08-15"
        ),
    )

    assert result.validation.sample_count == 2

    assert result.validation.accuracy == 0.0

    assert result.validation.recall == 0.0


def test_test_metrics_are_computed():
    df = make_feature_data()

    result = evaluate_majority_baseline(
        df,
        train_end=pd.Timestamp(
            "2026-08-10"
        ),
        validation_end=pd.Timestamp(
            "2026-08-15"
        ),
    )

    assert result.test.sample_count == 2

    assert result.test.accuracy == 0.5

    assert result.test.recall == 1.0


def test_baseline_evaluation_does_not_use_features():
    df = make_feature_data()

    # Replace every feature with nonsense values.
    for column in DEFAULT_FEATURE_COLUMNS:
        df[column] = -999999

    result = evaluate_majority_baseline(
        df,
        train_end=pd.Timestamp(
            "2026-08-10"
        ),
        validation_end=pd.Timestamp(
            "2026-08-15"
        ),
    )

    assert result.model.majority_class == 1
    assert result.validation.sample_count == 2
    assert result.test.sample_count == 2

def test_baseline_reports_benchmark_quality():
    df = make_feature_data()

    result = evaluate_majority_baseline(
        df,
        train_end=pd.Timestamp(
            "2026-08-10"
        ),
        validation_end=pd.Timestamp(
            "2026-08-15"
        ),
    )

    assert result.benchmark_quality.is_suitable is False

def test_benchmark_quality_contains_class_counts():
    df = make_feature_data()

    result = evaluate_majority_baseline(
        df,
        train_end=pd.Timestamp(
            "2026-08-10"
        ),
        validation_end=pd.Timestamp(
            "2026-08-15"
        ),
    )

    quality = result.benchmark_quality

    assert quality.validation_class_count == 1
    assert quality.test_class_count == 2
    assert quality.is_suitable is False