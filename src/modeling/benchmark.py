from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BenchmarkDataQuality:
    validation_class_count: int
    test_class_count: int
    validation_samples: int
    test_samples: int
    is_suitable: bool


def assess_benchmark_data(
    validation_y: pd.Series,
    test_y: pd.Series,
) -> BenchmarkDataQuality:
    """
    Assess whether validation and test targets contain
    enough class diversity for a meaningful binary
    classification benchmark.
    """

    if validation_y.empty:
        raise ValueError(
            "Validation target cannot be empty."
        )

    if test_y.empty:
        raise ValueError(
            "Test target cannot be empty."
        )

    validation_classes = (
        validation_y.dropna().nunique()
    )

    test_classes = (
        test_y.dropna().nunique()
    )

    return BenchmarkDataQuality(
        validation_class_count=int(
            validation_classes
        ),
        test_class_count=int(
            test_classes
        ),
        validation_samples=len(validation_y),
        test_samples=len(test_y),
        is_suitable=(
            validation_classes >= 2
            and test_classes >= 2
        ),
    )