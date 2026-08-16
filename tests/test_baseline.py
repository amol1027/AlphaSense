import pandas as pd
import pytest

from src.modeling.baseline import (
    BaselineModel,
    fit_majority_baseline,
)


def test_majority_baseline_learns_class_zero():
    y_train = pd.Series(
        [0, 0, 0, 1, 1],
        name="target_direction",
    )

    model = fit_majority_baseline(
        y_train
    )

    assert isinstance(
        model,
        BaselineModel,
    )

    assert model.majority_class == 0


def test_majority_baseline_learns_class_one():
    y_train = pd.Series(
        [1, 1, 1, 0],
        name="target_direction",
    )

    model = fit_majority_baseline(
        y_train
    )

    assert model.majority_class == 1


def test_majority_baseline_predicts_same_class():
    y_train = pd.Series(
        [1, 1, 1, 0],
    )

    X = pd.DataFrame(
        {
            "close": [
                100,
                101,
                102,
            ],
        }
    )

    model = fit_majority_baseline(
        y_train
    )

    predictions = model.predict(X)

    assert list(predictions) == [
        1,
        1,
        1,
    ]


def test_prediction_length_matches_input():
    y_train = pd.Series(
        [0, 0, 1],
    )

    X = pd.DataFrame(
        {
            "close": range(10),
        }
    )

    model = fit_majority_baseline(
        y_train
    )

    predictions = model.predict(X)

    assert len(predictions) == len(X)


def test_prediction_index_matches_input():
    y_train = pd.Series(
        [1, 1, 0],
    )

    X = pd.DataFrame(
        {
            "close": [100, 101],
            "volume": [1000, 2000],
        },
        index=[10, 20],
    )

    model = fit_majority_baseline(
        y_train
    )

    predictions = model.predict(X)

    assert list(
        predictions.index
    ) == [10, 20]


def test_empty_training_target_is_rejected():
    y_train = pd.Series(
        dtype=int,
    )

    with pytest.raises(
        ValueError,
        match="Training target cannot be empty",
    ):
        fit_majority_baseline(
            y_train
        )


def test_all_missing_training_target_is_rejected():
    y_train = pd.Series(
        [None, None],
        dtype="float64",
    )

    with pytest.raises(
        ValueError,
        match="no valid labels",
    ):
        fit_majority_baseline(
            y_train
        )


def test_tied_classes_choose_zero():
    y_train = pd.Series(
        [0, 0, 1, 1],
    )

    model = fit_majority_baseline(
        y_train
    )

    assert model.majority_class == 0