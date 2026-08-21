from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.build_features import (
    build_hourly_features,
)
from src.modeling.dataset import (
    build_train_validation_test,
)
from src.modeling.evaluation import (
    evaluate_classifier,
)
from src.modeling.logistic import (
    fit_logistic_model,
)


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


TRAIN_END = pd.Timestamp(
    "2026-07-25",
    tz="UTC",
)

VALIDATION_END = pd.Timestamp(
    "2026-08-02",
    tz="UTC",
)


MARKET_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

NEWS_FEATURES = [
    "sentiment_mean",
    "sentiment_std",
    "news_count",
    "positive_ratio",
    "negative_ratio",
]

REDDIT_FEATURES = [
    "reddit_sentiment_mean",
    "reddit_sentiment_std",
    "reddit_count",
    "reddit_positive_ratio",
    "reddit_negative_ratio",
    "reddit_score_mean",
    "reddit_comments_mean",
    "reddit_engagement_mean",
]


EXPERIMENTS = {
    "Market only": MARKET_FEATURES,
    "Market + News": (
        MARKET_FEATURES
        + NEWS_FEATURES
    ),
    "Market + Reddit": (
        MARKET_FEATURES
        + REDDIT_FEATURES
    ),
    "Market + News + Reddit": (
        MARKET_FEATURES
        + NEWS_FEATURES
        + REDDIT_FEATURES
    ),
}


def evaluate_asset_experiment(
    asset_features: pd.DataFrame,
    feature_columns: list[str],
):
    datasets = build_train_validation_test(
        asset_features,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
        feature_columns=feature_columns,
    )

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

    validation_metrics = (
        evaluate_classifier(
            datasets.validation.y,
            validation_predictions,
        )
    )

    test_metrics = (
        evaluate_classifier(
            datasets.test.y,
            test_predictions,
        )
    )

    return (
        validation_metrics,
        test_metrics,
        len(datasets.train.X),
        len(datasets.validation.X),
        len(datasets.test.X),
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

    print(
        f"Feature rows: {len(features)}"
    )

    print(
        "Assets:",
        features["asset"]
        .value_counts()
        .to_dict(),
    )

    results = []

    for asset in sorted(
        features["asset"].unique()
    ):
        print(
            f"\n{'=' * 60}"
        )
        print(
            f"ASSET: {asset}"
        )
        print(
            f"{'=' * 60}"
        )

        asset_features = features[
            features["asset"] == asset
        ].copy()

        print(
            "Feature rows:",
            len(asset_features),
        )

        for (
            experiment,
            feature_columns,
        ) in EXPERIMENTS.items():

            print(
                f"\nRunning: {experiment}"
            )

            (
                validation,
                test,
                train_samples,
                validation_samples,
                test_samples,
            ) = evaluate_asset_experiment(
                asset_features,
                feature_columns,
            )

            results.append(
                {
                    "asset": asset,
                    "experiment": experiment,
                    "validation_accuracy": (
                        validation.accuracy
                    ),
                    "validation_balanced_accuracy": (
                        validation.balanced_accuracy
                    ),
                    "test_accuracy": (
                        test.accuracy
                    ),
                    "test_balanced_accuracy": (
                        test.balanced_accuracy
                    ),
                    "test_precision": (
                        test.precision
                    ),
                    "test_recall": (
                        test.recall
                    ),
                    "train_samples": (
                        train_samples
                    ),
                    "validation_samples": (
                        validation_samples
                    ),
                    "test_samples": (
                        test_samples
                    ),
                }
            )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n\n=== ASSET-LEVEL ABLATION RESULTS ==="
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()