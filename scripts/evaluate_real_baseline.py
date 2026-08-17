from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.features.build_features import (
    build_hourly_features,
)
from src.modeling.baseline_runner import (
    evaluate_majority_baseline,
)


MARKET_PATH = (
    "data/raw/market/research_market_15m.csv"
)

NEWS_PATH = (
    "data/raw/news_sample.csv"
)

REDDIT_PATH = (
    "data/raw/reddit_sample.csv"
)

TRAIN_END = pd.Timestamp(
    "2026-07-25",
    tz="UTC",
)

VALIDATION_END = pd.Timestamp(
    "2026-08-02",
    tz="UTC",
)


def main():
    print("Building real feature dataset...")

    features = build_hourly_features(
        MARKET_PATH,
        NEWS_PATH,
        REDDIT_PATH,
    )

    print(
        f"Feature rows: {len(features)}"
    )

    print(
        "Assets:",
        features["asset"]
        .value_counts()
        .to_dict(),
    )

    print(
        "Prediction range:",
        features["prediction_timestamp"].min(),
        "→",
        features["prediction_timestamp"].max(),
    )

    print("\nEvaluating majority baseline...")

    result = evaluate_majority_baseline(
        features,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
    )

    print("\n=== REAL DATA MAJORITY BASELINE ===")

    print(
        "\nTraining majority class:",
        result.model.majority_class,
    )

    print("\nValidation")
    print(
        "  Accuracy:",
        result.validation.accuracy,
    )
    print(
        "  Balanced accuracy:",
        result.validation.balanced_accuracy,
    )
    print(
        "  Precision:",
        result.validation.precision,
    )
    print(
        "  Recall:",
        result.validation.recall,
    )
    print(
        "  Samples:",
        result.validation.sample_count,
    )
    print(
        "  Confusion matrix:",
        result.validation.confusion_matrix,
    )

    print("\nTest")
    print(
        "  Accuracy:",
        result.test.accuracy,
    )
    print(
        "  Balanced accuracy:",
        result.test.balanced_accuracy,
    )
    print(
        "  Precision:",
        result.test.precision,
    )
    print(
        "  Recall:",
        result.test.recall,
    )
    print(
        "  Samples:",
        result.test.sample_count,
    )
    print(
        "  Confusion matrix:",
        result.test.confusion_matrix,
    )

    print("\nBenchmark quality")
    print(
        "  Suitable:",
        result.benchmark_quality.is_suitable,
    )
    print(
        "  Validation classes:",
        result.benchmark_quality.validation_class_count,
    )
    print(
        "  Test classes:",
        result.benchmark_quality.test_class_count,
    )
    print(
        "  Validation samples:",
        result.benchmark_quality.validation_samples,
    )
    print(
        "  Test samples:",
        result.benchmark_quality.test_samples,
    )


if __name__ == "__main__":
    main()