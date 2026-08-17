from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LogisticModel:
    scaler: StandardScaler
    classifier: LogisticRegression
    feature_columns: tuple[str, ...]

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Predict binary direction for observations."""

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

        features = X[
            list(self.feature_columns)
        ]

        scaled = self.scaler.transform(features)

        predictions = self.classifier.predict(
            scaled
        )

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
            dtype=int,
        )


def fit_logistic_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticModel:
    """
    Fit Logistic Regression using training data only.

    Scaling parameters and classifier parameters are
    learned exclusively from X_train and y_train.
    """

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

    if y_train.nunique() < 2:
        raise ValueError(
            "Training target must contain both classes."
        )

    feature_columns = tuple(X_train.columns)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    classifier.fit(
        X_scaled,
        y_train,
    )

    return LogisticModel(
        scaler=scaler,
        classifier=classifier,
        feature_columns=feature_columns,
    )