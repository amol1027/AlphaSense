from dataclasses import dataclass

import pandas as pd

from src.modeling.baseline import (
    BaselineModel,
    fit_majority_baseline,
)
from src.modeling.dataset import (
    ModelingDatasetSplit,
    build_train_validation_test,
)
from src.modeling.evaluation import (
    ClassificationMetrics,
    evaluate_classifier,
)
from src.modeling.benchmark import (
    BenchmarkDataQuality,
    assess_benchmark_data,
)


@dataclass(frozen=True)
class BaselineEvaluation:
    model: BaselineModel
    validation: ClassificationMetrics
    test: ClassificationMetrics
    benchmark_quality: BenchmarkDataQuality

def evaluate_majority_baseline(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    feature_columns: list[str] | None = None,
) -> BaselineEvaluation:
    """
    Fit the majority baseline using training labels
    and evaluate it independently on validation and test.
    """

   
    datasets: ModelingDatasetSplit = (
        build_train_validation_test(
            df,
            train_end=train_end,
            validation_end=validation_end,
            feature_columns=feature_columns,
        )
    )

    model = fit_majority_baseline(
        datasets.train.y
    )

    validation_predictions = model.predict(
        datasets.validation.X
    )

    test_predictions = model.predict(
        datasets.test.X
    )

    validation_metrics = evaluate_classifier(
        datasets.validation.y,
        validation_predictions,
    )

    test_metrics = evaluate_classifier(
        datasets.test.y,
        test_predictions,
    )
    benchmark_quality = assess_benchmark_data(
            datasets.validation.y,
            datasets.test.y,
        )

    return BaselineEvaluation(
        model=model,
        validation=validation_metrics,
        test=test_metrics,
        benchmark_quality=benchmark_quality,
    )