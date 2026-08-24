from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.modeling.evaluation import evaluate_classifier
from src.modeling.logistic import fit_logistic_model
from src.modeling.tree_models import fit_hist_gradient_boosting


FEATURES_PATH = (
    "data/processed/phase1_features.csv"
)

FINAL_TEST_START = pd.Timestamp(
    "2026-08-10 00:00:00",
    tz="UTC",
)


CANDIDATES = {
    "RELIANCE": {
        "name": (
            "Reduced Engineered Market "
            "+ Logistic Regression"
        ),
        "features": [
            "return_15m",
            "return_30m",
            "return_1h",
            "high_low_range",
            "volume_change",
        ],
        "fit": fit_logistic_model,
    },
    "TCS": {
        "name": (
            "Market + News "
            "+ HistGradientBoosting"
        ),
        "features": [
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
        ],
        "fit": fit_hist_gradient_boosting,
    },
}


def majority_predict(
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.Series:
    majority_class = int(
        y_train.value_counts().idxmax()
    )

    return pd.Series(
        majority_class,
        index=y_test.index,
        name="prediction",
        dtype=int,
    )


def evaluate_asset(
    features: pd.DataFrame,
    asset: str,
    config: dict,
):
    asset_df = features[
        features["asset"] == asset
    ].copy()

    asset_df = asset_df.sort_values(
        "prediction_timestamp"
    )

    train_df = asset_df[
        asset_df["prediction_timestamp"]
        < FINAL_TEST_START
    ].copy()

    test_df = asset_df[
        asset_df["prediction_timestamp"]
        >= FINAL_TEST_START
    ].copy()

    required = [
        "target_direction",
        *config["features"],
    ]

    train_df = train_df.dropna(
        subset=required
    )

    test_df = test_df.dropna(
        subset=required
    )

    X_train = train_df[
        config["features"]
    ]

    y_train = train_df[
        "target_direction"
    ].astype(int)

    X_test = test_df[
        config["features"]
    ]

    y_test = test_df[
        "target_direction"
    ].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            f"{asset}: training data contains "
            "only one target class."
        )

    if y_test.nunique() < 2:
        raise ValueError(
            f"{asset}: test data contains "
            "only one target class."
        )

    model = config["fit"](
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    metrics = evaluate_classifier(
        y_test,
        predictions,
    )

    baseline_predictions = majority_predict(
        y_train,
        y_test,
    )

    baseline_metrics = evaluate_classifier(
        y_test,
        baseline_predictions,
    )

    return (
        metrics,
        baseline_metrics,
        len(train_df),
        len(test_df),
        test_df[
            "prediction_timestamp"
        ].min(),
        test_df[
            "prediction_timestamp"
        ].max(),
        y_train.mean(),
        y_test.mean(),
    )


def main():
    print("=" * 70)
    print("PHASE 1 LOCKED EVALUATION")
    print("=" * 70)

    print(
        f"Input: {FEATURES_PATH}"
    )

    features = pd.read_csv(
        FEATURES_PATH
    )

    features[
        "prediction_timestamp"
    ] = pd.to_datetime(
        features[
            "prediction_timestamp"
        ],
        utc=True,
    )

    print(
        f"\nFeature rows: {len(features):,}"
    )

    print(
        "\nAssets:"
    )

    print(
        features["asset"]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinal test starts:",
        FINAL_TEST_START,
    )

    print(
        "Final test dates:",
        features.loc[
            features[
                "prediction_timestamp"
            ] >= FINAL_TEST_START,
            "prediction_timestamp",
        ].dt.date.nunique(),
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOCKED RESULTS"
    )

    print(
        "=" * 70
    )

    results = []

    for asset, config in CANDIDATES.items():

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"ASSET: {asset}"
        )

        print(
            "=" * 70
        )

        print(
            "Frozen candidate:",
            config["name"],
        )

        (
            metrics,
            baseline,
            train_samples,
            test_samples,
            test_start,
            test_end,
            train_positive_rate,
            test_positive_rate,
        ) = evaluate_asset(
            features,
            asset,
            config,
        )

        print(
            "\nCandidate"
        )

        print(
            "  Accuracy:",
            f"{metrics.accuracy:.4f}",
        )

        print(
            "  Balanced accuracy:",
            f"{metrics.balanced_accuracy:.4f}",
        )

        print(
            "  Precision:",
            f"{metrics.precision:.4f}",
        )

        print(
            "  Recall:",
            f"{metrics.recall:.4f}",
        )

        print(
            "  Samples:",
            metrics.sample_count,
        )

        print(
            "  Confusion matrix:",
            metrics.confusion_matrix,
        )

        print(
            "\nMajority baseline"
        )

        print(
            "  Accuracy:",
            f"{baseline.accuracy:.4f}",
        )

        print(
            "  Balanced accuracy:",
            f"{baseline.balanced_accuracy:.4f}",
        )

        print(
            "  Precision:",
            f"{baseline.precision:.4f}",
        )

        print(
            "  Recall:",
            f"{baseline.recall:.4f}",
        )

        print(
            "  Samples:",
            baseline.sample_count,
        )

        print(
            "  Confusion matrix:",
            baseline.confusion_matrix,
        )

        print(
            "\nClass rates"
        )

        print(
            "  Train positive rate:",
            f"{train_positive_rate:.4f}",
        )

        print(
            "  Test positive rate:",
            f"{test_positive_rate:.4f}",
        )

        results.append(
            {
                "asset": asset,
                "candidate": config["name"],
                "train_samples": train_samples,
                "test_samples": test_samples,
                "test_start": test_start,
                "test_end": test_end,
                "train_positive_rate": (
                    train_positive_rate
                ),
                "test_positive_rate": (
                    test_positive_rate
                ),
                "candidate_accuracy": (
                    metrics.accuracy
                ),
                "candidate_balanced_accuracy": (
                    metrics.balanced_accuracy
                ),
                "baseline_accuracy": (
                    baseline.accuracy
                ),
                "baseline_balanced_accuracy": (
                    baseline.balanced_accuracy
                ),
                "balanced_accuracy_delta": (
                    metrics.balanced_accuracy
                    - baseline.balanced_accuracy
                ),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "PHASE 1 LOCKED SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:.4f}"
            ),
        )
    )


if __name__ == "__main__":
    main()