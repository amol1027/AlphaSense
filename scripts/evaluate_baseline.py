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


MARKET_PATH = "data/raw/market_sample.csv"
NEWS_PATH = "data/raw/news_sample.csv"
REDDIT_PATH = "data/raw/reddit_sample.csv"


def main():
    features = build_hourly_features(
        MARKET_PATH,
        NEWS_PATH,
        REDDIT_PATH,
    )

    print("Feature rows:", len(features))
    print(
        "Date range:",
        features["prediction_timestamp"].min(),
        "→",
        features["prediction_timestamp"].max(),
    )

    result = evaluate_majority_baseline(
        features,
        train_end=pd.Timestamp(
            "2026-08-10 12:15"
        ),
        validation_end=pd.Timestamp(
            "2026-08-10 14:15"
        ),
    )

    print("\n=== MAJORITY BASELINE ===")

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
    quality = result.benchmark_quality

    print("\nBenchmark quality")
    print(
        "  Suitable:",
        quality.is_suitable,
    )
    print(
        "  Validation classes:",
        quality.validation_class_count,
    )
    print(
        "  Test classes:",
        quality.test_class_count,
    )

    if not quality.is_suitable:
        print(
            "\nWARNING:"
            " Benchmark is not suitable "
            "for predictive-performance claims."
        )

if __name__ == "__main__":
    main()