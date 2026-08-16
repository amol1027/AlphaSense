from dataclasses import dataclass
from src.modeling.splits import chronological_split
import pandas as pd


TARGET_COLUMN = "target_direction"

FORBIDDEN_FEATURE_COLUMNS = {
    "future_close",
    "target_return",
    "target_direction",
}


DEFAULT_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sentiment_mean",
    "sentiment_std",
    "news_count",
    "positive_ratio",
    "negative_ratio",
    "reddit_sentiment_mean",
    "reddit_sentiment_std",
    "reddit_count",
    "reddit_positive_ratio",
    "reddit_negative_ratio",
    "reddit_score_mean",
    "reddit_comments_mean",
    "reddit_engagement_mean",
]


@dataclass(frozen=True)
class ModelingDataset:
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class ModelingDatasetSplit:
    train: ModelingDataset
    validation: ModelingDataset
    test: ModelingDataset


def build_modeling_dataset(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> ModelingDataset:
    """
    Build model inputs X and target y.

    Future-derived columns are never allowed
    to become model features.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            "DataFrame must contain "
            "'target_direction'."
        )

    columns = (
        DEFAULT_FEATURE_COLUMNS
        if feature_columns is None
        else feature_columns
    )

    forbidden = (
        set(columns)
        & FORBIDDEN_FEATURE_COLUMNS
    )

    if forbidden:
        raise ValueError(
            "Forbidden target/future columns "
            f"cannot be model features: "
            f"{sorted(forbidden)}"
        )

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing model feature columns: "
            f"{missing}"
        )

    data = df.copy()

    data = data.dropna(
        subset=[TARGET_COLUMN]
    ).reset_index(drop=True)

    X = data[columns].copy()
    y = data[TARGET_COLUMN].copy()

    return ModelingDataset(
        X=X,
        y=y,
    )


def build_modeling_dataset_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> ModelingDatasetSplit:
    """
    Build independent X/y datasets for train,
    validation, and test partitions.

    The chronological partitioning must already have
    happened before this function is called.
    """

    train = build_modeling_dataset(
        train_df,
        feature_columns=feature_columns,
    )

    validation = build_modeling_dataset(
        validation_df,
        feature_columns=feature_columns,
    )

    test = build_modeling_dataset(
        test_df,
        feature_columns=feature_columns,
    )

    return ModelingDatasetSplit(
        train=train,
        validation=validation,
        test=test,
    )

def build_train_validation_test(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    feature_columns: list[str] | None = None,
    require_non_empty: bool = True,
) -> ModelingDatasetSplit:
    """
    Build leakage-safe train, validation, and test
    modeling datasets.

    The chronological split happens before X/y
    construction.
    """

    split = chronological_split(
        df,
        train_end=train_end,
        validation_end=validation_end,
        require_non_empty=require_non_empty,
    )

    return build_modeling_dataset_split(
        train_df=split.train,
        validation_df=split.validation,
        test_df=split.test,
        feature_columns=feature_columns,
    )