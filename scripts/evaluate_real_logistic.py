from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.build_features import build_hourly_features
from src.modeling.dataset import build_train_validation_test
from src.modeling.evaluation import evaluate_classifier
from src.modeling.logistic import fit_logistic_model


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


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("  Accuracy:", metrics.accuracy)
    print(
        "  Balanced accuracy:",
        metrics.balanced_accuracy,
    )
    print("  Precision:", metrics.precision)
    print("  Recall:", metrics.recall)
    print("  Samples:", metrics.sample_count)
    print(
        "  Confusion matrix:",
        metrics.confusion_matrix,
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

    datasets = build_train_validation_test(
        features,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
    )

    print("\nModeling dataset")
    print(
        "  Train:",
        len(datasets.train.X),
    )
    print(
        "  Validation:",
        len(datasets.validation.X),
    )
    print(
        "  Test:",
        len(datasets.test.X),
    )

    print("\nFitting Logistic Regression...")

    model = fit_logistic_model(
        datasets.train.X,
        datasets.train.y,
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

    print("\n=== REAL DATA LOGISTIC REGRESSION ===")

    print_metrics(
        "Validation",
        validation_metrics,
    )

    print_metrics(
        "Test",
        test_metrics,
    )
    test_results = datasets.test.X.copy()

    test_results["asset"] = features.loc[
        datasets.test.X.index,
        "asset",
    ]

    test_results["actual"] = (
        datasets.test.y
    )

    test_results["prediction"] = (
        test_predictions
    )

    print("\n=== TEST RESULTS BY ASSET ===")

    for asset, group in test_results.groupby(
        "asset"
    ):
        metrics = evaluate_classifier(
            group["actual"],
            group["prediction"],
        )

        print_metrics(
            asset,
            metrics,
        )


if __name__ == "__main__":
    main()