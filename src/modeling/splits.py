from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    require_non_empty: bool = True,
) -> DatasetSplit:
    """
    Split feature data chronologically.

    Contract:

        timestamp < train_end
            -> train

        train_end <= timestamp < validation_end
            -> validation

        timestamp >= validation_end
            -> test

    The input DataFrame is never shuffled.
    """

    if "prediction_timestamp" not in df.columns:
        raise ValueError(
            "DataFrame must contain "
            "'prediction_timestamp'."
        )

    if train_end >= validation_end:
        raise ValueError(
            "train_end must be earlier "
            "than validation_end."
        )

    data = df.copy()

    data["prediction_timestamp"] = pd.to_datetime(
        data["prediction_timestamp"]
    )

    data = data.sort_values(
        "prediction_timestamp"
    ).reset_index(drop=True)

    train = data[
        data["prediction_timestamp"] < train_end
    ].copy()

    validation = data[
        (
            data["prediction_timestamp"]
            >= train_end
        )
        & (
            data["prediction_timestamp"]
            < validation_end
        )
    ].copy()

    test = data[
        data["prediction_timestamp"]
        >= validation_end
    ].copy()

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if require_non_empty:
        if train.empty:
            raise ValueError(
                "Training split is empty."
            )

        if validation.empty:
            raise ValueError(
                "Validation split is empty."
            )

        if test.empty:
            raise ValueError(
                "Test split is empty."
            )

    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )