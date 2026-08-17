import pandas as pd
import pytest

from src.modeling.logistic import (
    LogisticModel,
    fit_logistic_model,
)


def make_training_data():
    X = pd.DataFrame(
        {
            "close": [
                100,
                101,
                102,
                110,
                111,
                112,
            ],
            "volume": [
                1000,
                1100,
                1200,
                2000,
                2100,
                2200,
            ],
        }
    )

    y = pd.Series(
        [0, 0, 0, 1, 1, 1],
        name="target_direction",
    )

    return X, y


def test_logistic_model_fits():
    X, y = make_training_data()

    model = fit_logistic_model(
        X,
        y,
    )

    assert isinstance(
        model,
        LogisticModel,
    )


def test_logistic_model_predicts_binary_classes():
    X, y = make_training_data()

    model = fit_logistic_model(
        X,
        y,
    )

    predictions = model.predict(X)

    assert set(predictions).issubset(
        {0, 1}
    )


def test_prediction_length_matches_input():
    X, y = make_training_data()

    model = fit_logistic_model(
        X,
        y,
    )

    prediction_X = X.iloc[:3].copy()

    predictions = model.predict(
        prediction_X
    )

    assert len(predictions) == 3


def test_prediction_index_matches_input():
    X, y = make_training_data()

    model = fit_logistic_model(
        X,
        y,
    )

    prediction_X = X.iloc[:2].copy()
    prediction_X.index = [10, 20]

    predictions = model.predict(
        prediction_X
    )

    assert list(predictions.index) == [
        10,
        20,
    ]


def test_missing_feature_is_rejected():
    X, y = make_training_data()

    model = fit_logistic_model(
        X,
        y,
    )

    prediction_X = X.drop(
        columns=["volume"]
    )

    with pytest.raises(
        ValueError,
        match="Missing model feature columns",
    ):
        model.predict(prediction_X)


def test_empty_training_features_are_rejected():
    X, y = make_training_data()

    X = X.iloc[:0]

    with pytest.raises(
        ValueError,
        match="Training features cannot be empty",
    ):
        fit_logistic_model(X, y.iloc[:0])


def test_single_class_training_is_rejected():
    X, _ = make_training_data()

    y = pd.Series(
        [0, 0, 0, 0, 0, 0],
        name="target_direction",
    )

    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        fit_logistic_model(X, y)