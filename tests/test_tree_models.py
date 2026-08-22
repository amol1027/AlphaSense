import pandas as pd
import pytest

from src.modeling.tree_models import (
    fit_hist_gradient_boosting,
    fit_random_forest,
)


@pytest.fixture
def training_data():
    X = pd.DataFrame(
        {
            "feature_a": [
                0.1,
                0.2,
                0.3,
                0.8,
                0.9,
                1.0,
            ],
            "feature_b": [
                1.0,
                1.1,
                0.9,
                2.0,
                2.1,
                1.9,
            ],
        }
    )

    y = pd.Series(
        [0, 0, 0, 1, 1, 1],
        name="target_direction",
    )

    return X, y


def test_random_forest_predicts(
    training_data,
):
    X, y = training_data

    model = fit_random_forest(
        X,
        y,
    )

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert predictions.dtype == int
    assert set(predictions.unique()).issubset(
        {0, 1}
    )


def test_hist_gradient_boosting_predicts(
    training_data,
):
    X, y = training_data

    model = fit_hist_gradient_boosting(
        X,
        y,
    )

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert predictions.dtype == int
    assert set(predictions.unique()).issubset(
        {0, 1}
    )


def test_tree_model_rejects_missing_feature(
    training_data,
):
    X, y = training_data

    model = fit_random_forest(
        X,
        y,
    )

    with pytest.raises(ValueError):
        model.predict(
            X.drop(
                columns=["feature_a"]
            )
        )


def test_tree_model_rejects_nan_training_data(
    training_data,
):
    X, y = training_data

    X.loc[0, "feature_a"] = float("nan")

    with pytest.raises(ValueError):
        fit_random_forest(
            X,
            y,
        )