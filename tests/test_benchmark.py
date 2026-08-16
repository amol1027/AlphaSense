import pandas as pd
import pytest

from src.modeling.benchmark import (
    BenchmarkDataQuality,
    assess_benchmark_data,
)


def test_balanced_binary_benchmark_is_suitable():
    validation_y = pd.Series([0, 1, 0, 1])
    test_y = pd.Series([0, 1, 1, 0])

    result = assess_benchmark_data(
        validation_y,
        test_y,
    )

    assert isinstance(
        result,
        BenchmarkDataQuality,
    )

    assert result.validation_class_count == 2
    assert result.test_class_count == 2
    assert result.validation_samples == 4
    assert result.test_samples == 4
    assert result.is_suitable is True


def test_validation_single_class_is_not_suitable():
    validation_y = pd.Series([1, 1, 1, 1])
    test_y = pd.Series([0, 1, 0, 1])

    result = assess_benchmark_data(
        validation_y,
        test_y,
    )

    assert result.validation_class_count == 1
    assert result.test_class_count == 2
    assert result.is_suitable is False


def test_test_single_class_is_not_suitable():
    validation_y = pd.Series([0, 1, 0, 1])
    test_y = pd.Series([1, 1, 1, 1])

    result = assess_benchmark_data(
        validation_y,
        test_y,
    )

    assert result.validation_class_count == 2
    assert result.test_class_count == 1
    assert result.is_suitable is False


def test_empty_validation_is_rejected():
    validation_y = pd.Series(
        dtype=int
    )
    test_y = pd.Series([0, 1])

    with pytest.raises(
        ValueError,
        match="Validation target cannot be empty",
    ):
        assess_benchmark_data(
            validation_y,
            test_y,
        )


def test_empty_test_is_rejected():
    validation_y = pd.Series([0, 1])
    test_y = pd.Series(
        dtype=int
    )

    with pytest.raises(
        ValueError,
        match="Test target cannot be empty",
    ):
        assess_benchmark_data(
            validation_y,
            test_y,
        )


def test_missing_labels_are_ignored():
    validation_y = pd.Series(
        [0, 1, None]
    )

    test_y = pd.Series(
        [0, 1, None]
    )

    result = assess_benchmark_data(
        validation_y,
        test_y,
    )

    assert result.validation_class_count == 2
    assert result.test_class_count == 2
    assert result.is_suitable is True