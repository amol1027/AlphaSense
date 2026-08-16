from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BaselineModel:
    majority_class: int

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict the training-set majority class
        for every observation.
        """

        return pd.Series(
            self.majority_class,
            index=X.index,
            name="prediction",
            dtype=int,
        )


def fit_majority_baseline(
    y_train: pd.Series,
) -> BaselineModel:
    """
    Fit a majority-class baseline using training
    labels only.
    """

    if y_train.empty:
        raise ValueError(
            "Training target cannot be empty."
        )

    values = y_train.dropna()

    if values.empty:
        raise ValueError(
            "Training target contains no valid labels."
        )

    counts = values.value_counts()

    majority_class = int(
        counts.sort_index()
        .idxmax()
    )

    return BaselineModel(
        majority_class=majority_class,
    )