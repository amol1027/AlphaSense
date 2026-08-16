import pandas as pd
import pytest

from src.modeling.dataset import (
    DEFAULT_FEATURE_COLUMNS,
    ModelingDataset,
    ModelingDatasetSplit,
    build_modeling_dataset,
    build_modeling_dataset_split,
    build_train_validation_test,
)


def make_feature_data():
    return pd.DataFrame(
        {
            "asset": [
                "TCS",
                "RELIANCE",
                "TCS",
            ],
            "exchange": [
                "NSE",
                "NSE",
                "NSE",
            ],
            "prediction_timestamp": pd.to_datetime(
                [
                    "2026-08-10 10:15",
                    "2026-08-10 10:15",
                    "2026-08-10 11:15",
                ]
            ),
            "open": [3500, 1400, 3520],
            "high": [3525, 1415, 3535],
            "low": [3495, 1395, 3515],
            "close": [3520, 1410, 3530],
            "volume": [120000, 95000, 130000],
            "future_close": [
                3530,
                1408,
                3525,
            ],
            "target_return": [
                0.0028,
                -0.0014,
                -0.0014,
            ],
            "target_direction": [
                1,
                0,
                0,
            ],
            "sentiment_mean": [
                0.375,
                0.750,
                0.150,
            ],
            "sentiment_std": [
                0.530,
                0.000,
                0.627,
            ],
            "news_count": [
                2,
                1,
                5,
            ],
            "positive_ratio": [
                0.5,
                1.0,
                0.4,
            ],
            "negative_ratio": [
                0.0,
                0.0,
                0.2,
            ],
            "reddit_sentiment_mean": [
                0.75,
                0.0,
                0.45,
            ],
            "reddit_sentiment_std": [
                0.0,
                0.0,
                0.41,
            ],
            "reddit_count": [
                2,
                1,
                5,
            ],
            "reddit_positive_ratio": [
                1.0,
                0.0,
                0.6,
            ],
            "reddit_negative_ratio": [
                0.0,
                0.0,
                0.0,
            ],
            "reddit_score_mean": [
                32.5,
                35.0,
                46.0,
            ],
            "reddit_comments_mean": [
                8.0,
                8.0,
                14.2,
            ],
            "reddit_engagement_mean": [
                40.5,
                43.0,
                60.2,
            ],
        }
    )


def test_build_modeling_dataset():
    df = make_feature_data()

    result = build_modeling_dataset(df)

    assert isinstance(
        result,
        ModelingDataset,
    )

    assert list(result.X.columns) == (
        DEFAULT_FEATURE_COLUMNS
    )

    assert len(result.X) == 3
    assert len(result.y) == 3


def test_future_columns_are_not_features():
    df = make_feature_data()

    result = build_modeling_dataset(df)

    assert "future_close" not in result.X.columns
    assert "target_return" not in result.X.columns
    assert "target_direction" not in result.X.columns


def test_target_is_returned_separately():
    df = make_feature_data()

    result = build_modeling_dataset(df)

    assert list(result.y) == [1, 0, 0]
    assert result.y.name == "target_direction"


def test_metadata_is_not_default_feature():
    df = make_feature_data()

    result = build_modeling_dataset(df)

    assert "asset" not in result.X.columns
    assert "exchange" not in result.X.columns
    assert (
        "prediction_timestamp"
        not in result.X.columns
    )


def test_missing_target_is_rejected():
    df = make_feature_data().drop(
        columns=["target_direction"]
    )

    with pytest.raises(
        ValueError,
        match="target_direction",
    ):
        build_modeling_dataset(df)


def test_missing_feature_is_rejected():
    df = make_feature_data().drop(
        columns=["volume"]
    )

    with pytest.raises(
        ValueError,
        match="Missing model feature columns",
    ):
        build_modeling_dataset(df)


def test_forbidden_future_feature_is_rejected():
    df = make_feature_data()

    with pytest.raises(
        ValueError,
        match="Forbidden target/future columns",
    ):
        build_modeling_dataset(
            df,
            feature_columns=[
                "close",
                "future_close",
            ],
        )


def test_rows_without_target_are_removed():
    df = make_feature_data()

    df.loc[
        2,
        "target_direction",
    ] = None

    result = build_modeling_dataset(df)

    assert len(result.X) == 2
    assert len(result.y) == 2
def test_build_modeling_dataset_split():
    df = make_feature_data()

    train_df = df.iloc[[0]].copy()
    validation_df = df.iloc[[1]].copy()
    test_df = df.iloc[[2]].copy()

    result = build_modeling_dataset_split(
        train_df,
        validation_df,
        test_df,
    )

    assert isinstance(
        result,
        ModelingDatasetSplit,
    )

    assert len(result.train.X) == 1
    assert len(result.validation.X) == 1
    assert len(result.test.X) == 1

    assert len(result.train.y) == 1
    assert len(result.validation.y) == 1
    assert len(result.test.y) == 1
def test_future_columns_are_excluded_from_all_splits():
    df = make_feature_data()

    result = build_modeling_dataset_split(
        df.iloc[[0]],
        df.iloc[[1]],
        df.iloc[[2]],
    )

    for dataset in [
        result.train,
        result.validation,
        result.test,
    ]:
        assert "future_close" not in dataset.X
        assert "target_return" not in dataset.X
        assert "target_direction" not in dataset.X

def test_each_split_keeps_its_own_target():
    df = make_feature_data()

    result = build_modeling_dataset_split(
        df.iloc[[0]],
        df.iloc[[1]],
        df.iloc[[2]],
    )

    assert result.train.y.iloc[0] == 1
    assert result.validation.y.iloc[0] == 0
    assert result.test.y.iloc[0] == 0

def test_missing_target_rows_are_removed_independently():
    df = make_feature_data()

    train_df = df.iloc[[0]].copy()

    validation_df = df.iloc[[1]].copy()
    validation_df.loc[
        validation_df.index[0],
        "target_direction",
    ] = None

    test_df = df.iloc[[2]].copy()

    result = build_modeling_dataset_split(
        train_df,
        validation_df,
        test_df,
    )

    assert len(result.train.X) == 1
    assert len(result.validation.X) == 0
    assert len(result.test.X) == 1

def test_build_train_validation_test():
    df = make_feature_data().copy()

    df["prediction_timestamp"] = pd.to_datetime(
        [
            "2026-08-05 10:15",
            "2026-08-12 10:15",
            "2026-08-20 10:15",
        ]
    )

    extra = df.copy()

    extra["prediction_timestamp"] = pd.to_datetime(
        [
            "2026-08-06 10:15",
            "2026-08-13 10:15",
            "2026-08-21 10:15",
        ]
    )

    df = pd.concat(
        [df, extra],
        ignore_index=True,
    )

    result = build_train_validation_test(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    assert isinstance(
        result,
        ModelingDatasetSplit,
    )

    assert len(result.train.X) == 2
    assert len(result.validation.X) == 2
    assert len(result.test.X) == 2
    
def test_end_to_end_pipeline_excludes_future_columns():
    df = make_feature_data()

    extra = df.copy()

    extra["prediction_timestamp"] = pd.to_datetime(
        [
            "2026-08-05 10:15",
            "2026-08-12 10:15",
            "2026-08-20 10:15",
        ]
    )

    df = pd.concat(
        [df, extra],
        ignore_index=True,
    )

    result = build_train_validation_test(
        df,
        train_end=pd.Timestamp(
            "2026-08-10"
        ),
        validation_end=pd.Timestamp(
            "2026-08-15"
        ),
    )

    for dataset in [
        result.train,
        result.validation,
        result.test,
    ]:
        assert "future_close" not in dataset.X.columns
        assert "target_return" not in dataset.X.columns
        assert "target_direction" not in dataset.X.columns

def test_end_to_end_pipeline_preserves_time_boundaries():
    df = make_feature_data().copy()

    df["prediction_timestamp"] = pd.to_datetime(
        [
            "2026-08-05 10:15",
            "2026-08-12 10:15",
            "2026-08-20 10:15",
        ]
    )

    extra = df.copy()

    extra["prediction_timestamp"] = pd.to_datetime(
        [
            "2026-08-06 10:15",
            "2026-08-13 10:15",
            "2026-08-21 10:15",
        ]
    )

    df = pd.concat(
        [df, extra],
        ignore_index=True,
    )

    result = build_train_validation_test(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    assert len(result.train.y) == 2
    assert len(result.validation.y) == 2
    assert len(result.test.y) == 2