from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.build_features import build_hourly_features
from src.modeling.evaluation import evaluate_classifier
from src.modeling.logistic import fit_logistic_model
from src.modeling.tree_models import fit_hist_gradient_boosting


MARKET_PATH = (
    "data/raw/market/research_market_15m.csv"
)

NEWS_PATH = (
    "data/raw/news/research_news.csv"
)

REDDIT_PATH = (
    "data/raw/reddit_sample.csv"
)

NEWS_SENTIMENT_PATH = (
    "data/processed/research_news_sentiment.csv"
)


# Frozen candidates — DO NOT TUNE HERE.
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
    final_test_start: pd.Timestamp,
):
    asset_df = features[
        features["asset"] == asset
    ].copy()

    asset_df = asset_df.sort_values(
        "prediction_timestamp"
    )

    train_df = asset_df[
        asset_df["prediction_timestamp"]
        < final_test_start
    ].copy()

    test_df = asset_df[
        asset_df["prediction_timestamp"]
        >= final_test_start
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
    ]

    X_test = test_df[
        config["features"]
    ]

    y_test = test_df[
        "target_direction"
    ]

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
    )


def main():

    print(
        "Building real feature dataset..."
    )

    features = build_hourly_features(
        MARKET_PATH,
        NEWS_PATH,
        REDDIT_PATH,
        news_sentiment_path=(
            NEWS_SENTIMENT_PATH
        ),
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
        f"Feature rows: {len(features)}"
    )

    print(
        "Assets:",
        features["asset"]
        .value_counts()
        .to_dict(),
    )

    # Final untouched period.
    #
    # Everything before this timestamp is used
    # for fitting. Everything from this timestamp
    # onward is evaluated once.
    final_test_start = pd.Timestamp(
        "2026-08-03 00:00:00",
        tz="UTC",
    )

    print(
        "\nFinal test starts:",
        final_test_start,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL LOCKED EVALUATION"
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
        ) = evaluate_asset(
            features,
            asset,
            config,
            final_test_start,
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

        results.append(
            {
                "asset": asset,
                "candidate": config["name"],
                "train_samples": train_samples,
                "test_samples": test_samples,
                "test_start": test_start,
                "test_end": test_end,
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

    results_df = pd.DataFrame(results)

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "FINAL LOCKED SUMMARY"
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