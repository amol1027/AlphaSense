import pandas as pd
import pytest

from src.modeling.evaluation import (
    ClassificationMetrics,
    evaluate_classifier,
)


def test_evaluate_classifier_returns_metrics():
    y_true = pd.Series(
        [0, 0, 1, 1],
    )

    y_pred = pd.Series(
        [0, 1, 1, 1],
    )

    result = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert isinstance(
        result,
        ClassificationMetrics,
    )


def test_accuracy():
    y_true = pd.Series(
        [0, 0, 1, 1],
    )

    y_pred = pd.Series(
        [0, 1, 1, 1],
    )

    result = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert result.accuracy == 0.75


def test_balanced_accuracy():
    y_true = pd.Series(
        [0, 0, 1, 1],
    )

    y_pred = pd.Series(
        [0, 1, 1, 1],
    )

    result = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert result.balanced_accuracy == 0.75


def test_precision():
    y_true = pd.Series(
        [0, 0, 1, 1],
    )

    y_pred = pd.Series(
        [0, 1, 1, 1],
    )

    result = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert result.precision == pytest.approx(
        2 / 3
    )


def test_recall():
    y_true = pd.Series(
        [0, 0, 1, 1],
    )

    y_pred = pd.Series(
        [0, 1, 1, 1],
    )

    result = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert result.recall == 1.0


def test_confusion_matrix():
    y_true = pd.Series(
        [0, 0, 1, 1],
    )

    y_pred = pd.Series(
        [0, 1, 1, 1],
    )

    result = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert result.confusion_matrix == (
        (1, 1),
        (0, 2),
    )


def test_sample_count():
    y_true = pd.Series(
        [0, 1, 0, 1, 1],
    )

    y_pred = pd.Series(
        [0, 1, 1, 1, 0],
    )

    result = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert result.sample_count == 5


def test_mismatched_lengths_are_rejected():
    y_true = pd.Series(
        [0, 1, 0],
    )

    y_pred = pd.Series(
        [0, 1],
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        evaluate_classifier(
            y_true,
            y_pred,
        )


def test_empty_predictions_are_rejected():
    y_true = pd.Series(
        dtype=int,
    )

    y_pred = pd.Series(
        dtype=int,
    )

    with pytest.raises(
        ValueError,
        match="empty predictions",
    ):
        evaluate_classifier(
            y_true,
            y_pred,
        )


def test_majority_prediction_does_not_crash():
    y_true = pd.Series(
        [0, 0, 0, 1, 1],
    )

    y_pred = pd.Series(
        [0, 0, 0, 0, 0],
    )

    result = evaluate_classifier(
        y_true,
        y_pred,
    )

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.confusion_matrix == (
        (3, 0),
        (2, 0),
    )