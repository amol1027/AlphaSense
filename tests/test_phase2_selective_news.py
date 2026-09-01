import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_phase2_selective_news import (
    NEWS_FEATURES,
    add_event_regime_features,
    apply_selective_news_policy,
    calculate_event_thresholds,
)


def make_news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "news_count": [0, 1, 3, 5, 10, 20],
            "news_burst": [np.nan, 0.5, 1.0, 2.0, 5.0, 10.0],
            "sentiment_std": [np.nan, 0.1, 0.2, 0.3, 0.4, 0.5],
            "positive_ratio": [np.nan, 0.1, 0.2, 0.3, 0.4, 0.5],
            "negative_ratio": [np.nan, 0.1, 0.2, 0.3, 0.4, 0.5],
            "sentiment_imbalance": [
                np.nan,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            "source_diversity": [
                np.nan,
                1,
                1,
                2,
                3,
                4,
            ],
        }
    )


def test_news_features_constant_is_not_required():
    assert "news_count" in NEWS_FEATURES
    assert "news_burst" in NEWS_FEATURES
    assert "sentiment_std" in NEWS_FEATURES


def test_event_thresholds_use_training_data_only():
    train = pd.DataFrame(
        {
            "news_count": [1, 2, 3, 4],
            "news_burst": [1.0, 2.0, 3.0, 4.0],
        }
    )

    thresholds = calculate_event_thresholds(train)

    assert thresholds["news_count"] == pytest.approx(3.25)
    assert thresholds["news_burst"] == pytest.approx(3.25)


def test_event_thresholds_do_not_use_oos_data():
    train = pd.DataFrame(
        {
            "news_count": [1, 2, 3, 4],
            "news_burst": [1.0, 2.0, 3.0, 4.0],
        }
    )

    oos = pd.DataFrame(
        {
            "news_count": [100, 200, 300],
            "news_burst": [100.0, 200.0, 300.0],
        }
    )

    train_thresholds = calculate_event_thresholds(train)

    combined = pd.concat([train, oos], ignore_index=True)
    combined_thresholds = calculate_event_thresholds(combined)

    assert (
        train_thresholds["news_count"]
        != combined_thresholds["news_count"]
    )

    assert (
        train_thresholds["news_burst"]
        != combined_thresholds["news_burst"]
    )


def test_event_regime_features_are_binary():
    df = make_news_frame()

    thresholds = calculate_event_thresholds(df)

    result = add_event_regime_features(
        df,
        thresholds,
    )

    assert "high_news_event" in result.columns
    assert "high_news_burst" in result.columns

    assert set(
        result["high_news_event"].dropna().unique()
    ).issubset({0, 1})

    assert set(
        result["high_news_burst"].dropna().unique()
    ).issubset({0, 1})


def test_high_news_event_requires_news_count_threshold():
    df = pd.DataFrame(
        {
            "news_count": [1, 2, 3, 4],
            "news_burst": [1, 1, 1, 1],
        }
    )

    thresholds = {
        "news_count": 3.0,
        "news_burst": 1.0,
    }

    result = add_event_regime_features(
        df,
        thresholds,
    )

    assert result["high_news_event"].tolist() == [
        0,
        0,
        1,
        1,
    ]


def test_high_news_burst_requires_burst_threshold():
    df = pd.DataFrame(
        {
            "news_count": [1, 1, 1, 1],
            "news_burst": [0.5, 1.0, 2.0, 3.0],
        }
    )

    thresholds = {
        "news_count": 1.0,
        "news_burst": 2.0,
    }

    result = add_event_regime_features(
        df,
        thresholds,
    )

    assert result["high_news_burst"].tolist() == [
        0,
        0,
        1,
        1,
    ]


def test_missing_news_is_not_treated_as_high_intensity():
    df = pd.DataFrame(
        {
            "news_count": [np.nan, 1, 5],
            "news_burst": [np.nan, np.nan, 5],
        }
    )

    thresholds = {
        "news_count": 3.0,
        "news_burst": 3.0,
    }

    result = add_event_regime_features(
        df,
        thresholds,
    )

    assert result.loc[0, "high_news_event"] == 0
    assert result.loc[0, "high_news_burst"] == 0


def test_selective_policy_only_uses_directional_news_signal():
    df = pd.DataFrame(
        {
            "prob_down": [0.1, 0.4, 0.2],
            "prob_neutral": [0.8, 0.3, 0.2],
            "prob_up": [0.1, 0.3, 0.6],
            "high_news_event": [0, 1, 1],
        }
    )

    result = apply_selective_news_policy(
        df,
        up_probability_threshold=0.5,
        down_probability_threshold=0.5,
    )

    assert result.tolist() == [
        "NEUTRAL",
        "NEUTRAL",
        "UP",
    ]


def test_selective_policy_requires_event_regime():
    df = pd.DataFrame(
        {
            "prob_down": [0.8, 0.8],
            "prob_neutral": [0.1, 0.1],
            "prob_up": [0.1, 0.1],
            "high_news_event": [0, 1],
        }
    )

    result = apply_selective_news_policy(
        df,
        up_probability_threshold=0.5,
        down_probability_threshold=0.5,
    )

    assert result.tolist() == [
        "NEUTRAL",
        "DOWN",
    ]

def test_phase27_development_window_constants():
    from scripts.evaluate_phase2_selective_news import (
        TRAIN_START,
        TRAIN_END,
        OOS_START,
        OOS_END,
        LOCKED_START,
    )

    assert TRAIN_START < TRAIN_END
    assert TRAIN_END == OOS_START
    assert OOS_START < OOS_END
    assert OOS_END == LOCKED_START

def test_phase27_locked_boundary_is_strict():
    from scripts.evaluate_phase2_selective_news import (
        LOCKED_START,
    )

    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2026-08-09 23:45:00+00:00",
                "2026-08-10 00:00:00+00:00",
            ],
            utc=True,
        )
    )

    development = timestamps[
        timestamps < LOCKED_START
    ]

    locked = timestamps[
        timestamps >= LOCKED_START
    ]

    assert len(development) == 1
    assert len(locked) == 1
    assert development.iloc[0] < LOCKED_START
    assert locked.iloc[0] >= LOCKED_START