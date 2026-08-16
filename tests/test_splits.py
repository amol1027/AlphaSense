import pandas as pd
import pytest

from src.modeling.splits import (
    DatasetSplit,
    chronological_split,
)


def make_sample_data():
    return pd.DataFrame(
        {
            "asset": [
                "TCS",
                "TCS",
                "TCS",
                "TCS",
                "TCS",
                "TCS",
            ],
            "prediction_timestamp": pd.to_datetime(
                [
                    "2026-08-01 10:15",
                    "2026-08-05 10:15",
                    "2026-08-09 10:15",
                    "2026-08-10 10:15",
                    "2026-08-15 10:15",
                    "2026-08-20 10:15",
                ]
            ),
            "target_direction": [
                1,
                0,
                1,
                1,
                0,
                1,
            ],
        }
    )

def make_multi_asset_data():
    return pd.DataFrame(
        {
            "asset": [
                "TCS",
                "TCS",
                "TCS",
                "TCS",
                "RELIANCE",
                "RELIANCE",
                "RELIANCE",
                "RELIANCE",
            ],
            "prediction_timestamp": pd.to_datetime(
                [
                    "2026-08-01 10:15",
                    "2026-08-05 10:15",
                    "2026-08-10 10:15",
                    "2026-08-20 10:15",
                    "2026-08-02 10:15",
                    "2026-08-09 10:15",
                    "2026-08-15 10:15",
                    "2026-08-25 10:15",
                ]
            ),
            "target_direction": [
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )

def test_multi_asset_split_uses_global_time_boundaries():
    df = make_multi_asset_data()

    result = chronological_split(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    assert set(result.train["asset"]) == {
        "TCS",
        "RELIANCE",
    }

    assert set(result.validation["asset"]) == {
        "TCS",
    }

    assert set(result.test["asset"]) == {
        "TCS",
        "RELIANCE",
    }

def test_multi_asset_train_contains_no_future_rows():
    df = make_multi_asset_data()

    result = chronological_split(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    assert (
        result.train["prediction_timestamp"]
        < pd.Timestamp("2026-08-10")
    ).all()

def test_multi_asset_validation_respects_global_boundaries():
    df = make_multi_asset_data()

    result = chronological_split(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    assert (
        result.validation["prediction_timestamp"]
        >= pd.Timestamp("2026-08-10")
    ).all()

    assert (
        result.validation["prediction_timestamp"]
        < pd.Timestamp("2026-08-15")
    ).all()

def test_multi_asset_test_contains_only_future_rows():
    df = make_multi_asset_data()

    result = chronological_split(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    assert (
        result.test["prediction_timestamp"]
        >= pd.Timestamp("2026-08-15")
    ).all()

def test_missing_asset_observations_do_not_change_global_boundaries():
    df = pd.DataFrame(
        {
            "asset": [
                "TCS",
                "TCS",
                "TCS",
                "RELIANCE",
                "RELIANCE",
            ],
            "prediction_timestamp": pd.to_datetime(
                [
                    "2026-08-01 10:15",
                    "2026-08-10 10:15",
                    "2026-08-20 10:15",
                    "2026-08-01 10:15",
                    "2026-08-20 10:15",
                ]
            ),
            "target_direction": [
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )

    result = chronological_split(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    assert len(result.train) == 2
    assert len(result.validation) == 1
    assert len(result.test) == 2

    assert set(result.train["asset"]) == {
        "TCS",
        "RELIANCE",
    }

    assert set(result.validation["asset"]) == {
        "TCS",
    }

    assert set(result.test["asset"]) == {
        "TCS",
        "RELIANCE",
    }


def test_chronological_split():
    df = make_sample_data()

    result = chronological_split(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    assert isinstance(result, DatasetSplit)

    assert len(result.train) == 3
    assert len(result.validation) == 1
    assert len(result.test) == 2


def test_split_boundaries_do_not_overlap():
    df = make_sample_data()

    result = chronological_split(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    train_times = set(
        result.train["prediction_timestamp"]
    )

    validation_times = set(
        result.validation["prediction_timestamp"]
    )

    test_times = set(
        result.test["prediction_timestamp"]
    )

    assert train_times.isdisjoint(validation_times)
    assert train_times.isdisjoint(test_times)
    assert validation_times.isdisjoint(test_times)


def test_split_is_chronological():
    df = make_sample_data()

    # Deliberately shuffle the input.
    df = df.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    result = chronological_split(
        df,
        train_end=pd.Timestamp("2026-08-10"),
        validation_end=pd.Timestamp("2026-08-15"),
    )

    assert result.train[
        "prediction_timestamp"
    ].is_monotonic_increasing

    assert result.validation[
        "prediction_timestamp"
    ].is_monotonic_increasing

    assert result.test[
        "prediction_timestamp"
    ].is_monotonic_increasing


def test_invalid_split_boundaries_are_rejected():
    df = make_sample_data()

    with pytest.raises(ValueError):
        chronological_split(
            df,
            train_end=pd.Timestamp("2026-08-15"),
            validation_end=pd.Timestamp("2026-08-10"),
        )


def test_missing_timestamp_column_is_rejected():
    df = pd.DataFrame(
        {
            "asset": ["TCS"],
            "target_direction": [1],
        }
    )

    with pytest.raises(ValueError):
        chronological_split(
            df,
            train_end=pd.Timestamp("2026-08-10"),
            validation_end=pd.Timestamp("2026-08-15"),
        )


def test_empty_validation_split_is_rejected():
    df = make_sample_data()

    with pytest.raises(
        ValueError,
        match="Validation split is empty",
    ):
        chronological_split(
            df,
            train_end=pd.Timestamp("2026-08-11"),
            validation_end=pd.Timestamp("2026-08-15"),
        )

def test_empty_test_split_is_rejected():
    df = make_sample_data()

    with pytest.raises(
        ValueError,
        match="Test split is empty",
    ):
        chronological_split(
            df,
            train_end=pd.Timestamp("2026-08-05"),
            validation_end=pd.Timestamp("2026-08-21"),
        )




def test_empty_split_can_be_allowed():
    df = make_sample_data()

    result = chronological_split(
        df,
        train_end=pd.Timestamp("2026-08-11"),
        validation_end=pd.Timestamp("2026-08-15"),
        require_non_empty=False,
    )

    assert result.validation.empty