from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    sample_count: int


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> ClassificationMetrics:
    """
    Evaluate binary classification predictions.

    The evaluator does not fit or modify a model.
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have "
            "the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot evaluate empty predictions."
        )

    y_true = pd.Series(y_true).reset_index(
        drop=True
    )
    y_pred = pd.Series(y_pred).reset_index(
        drop=True
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    if y_true.nunique() == 1:
        balanced_accuracy = float(
            accuracy_score(
                y_true,
                y_pred,
            )
        )
    else:
        balanced_accuracy = float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )

    return ClassificationMetrics(
        accuracy=float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        balanced_accuracy=balanced_accuracy,
        precision=float(
            precision_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="binary",
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="binary",
                zero_division=0,
            )
        ),
        confusion_matrix=(
            (
                int(matrix[0, 0]),
                int(matrix[0, 1]),
            ),
            (
                int(matrix[1, 0]),
                int(matrix[1, 1]),
            ),
        ),
        sample_count=len(y_true),
    )