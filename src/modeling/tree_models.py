from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)


@dataclass(frozen=True)
class TreeModel:
    classifier: object
    feature_columns: tuple[str, ...]

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Predict binary direction."""

        missing = [
            column
            for column in self.feature_columns
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                "Missing model feature columns: "
                f"{missing}"
            )

        predictions = self.classifier.predict(
            X[
                list(self.feature_columns)
            ]
        )

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
            dtype=int,
        )


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> TreeModel:
    """
    Fit Random Forest using training data only.
    """

    _validate_training_data(
        X_train,
        y_train,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    return TreeModel(
        classifier=classifier,
        feature_columns=tuple(
            X_train.columns
        ),
    )


def fit_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> TreeModel:
    """
    Fit Histogram Gradient Boosting using
    training data only.
    """

    _validate_training_data(
        X_train,
        y_train,
    )

    classifier = (
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    classifier.fit(
        X_train,
        y_train,
    )

    return TreeModel(
        classifier=classifier,
        feature_columns=tuple(
            X_train.columns
        ),
    )


def _validate_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    if X_train.empty:
        raise ValueError(
            "Training features cannot be empty."
        )

    if y_train.empty:
        raise ValueError(
            "Training target cannot be empty."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training features and target must "
            "have the same length."
        )

    if y_train.isna().any():
        raise ValueError(
            "Training target contains missing labels."
        )

    if X_train.isna().any().any():
        raise ValueError(
            "Training features contain missing values."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training target must contain both classes."
        )