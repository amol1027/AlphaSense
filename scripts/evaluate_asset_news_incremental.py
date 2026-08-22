from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.build_features import build_hourly_features
from src.modeling.evaluation import evaluate_classifier
from src.modeling.logistic import fit_logistic_model
from src.modeling.tree_models import (
    fit_random_forest,
    fit_hist_gradient_boosting,
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


RELIANCE_MARKET_FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "volume_change",
]

TCS_MARKET_FEATURES = [
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


FEATURE_SETS = {
    "RELIANCE": {
        "Market": RELIANCE_MARKET_FEATURES,
        "Market + News": (
            RELIANCE_MARKET_FEATURES
            + NEWS_FEATURES
        ),
    },
    "TCS": {
        "Market": TCS_MARKET_FEATURES,
        "Market + News": (
            TCS_MARKET_FEATURES
            + NEWS_FEATURES
        ),
    },
}


MODELS = {
    "Logistic Regression": fit_logistic_model,
    "Random Forest": fit_random_forest,
    "HistGradientBoosting": (
        fit_hist_gradient_boosting
    ),
}


MIN_TRAIN_DAYS = 10


def get_walk_forward_periods(
    features: pd.DataFrame,
):
    data = features.sort_values(
        "prediction_timestamp"
    ).reset_index(drop=True)

    data["date"] = (
        data["prediction_timestamp"]
        .dt.date
    )

    trading_dates = sorted(
        data["date"].unique()
    )

    periods = []

    for index in range(
        MIN_TRAIN_DAYS,
        len(trading_dates),
    ):
        test_date = trading_dates[index]

        train_dates = trading_dates[
            :index
        ]

        train_df = data[
            data["date"].isin(train_dates)
        ].copy()

        test_df = data[
            data["date"] == test_date
        ].copy()

        periods.append(
            (
                test_date,
                train_df,
                test_df,
            )
        )

    return periods


def evaluate_period(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
):
    required = [
        "target_direction"
    ] + feature_columns

    train_df = train_df.dropna(
        subset=required
    ).copy()

    test_df = test_df.dropna(
        subset=required
    ).copy()

    if train_df.empty or test_df.empty:
        return None

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "target_direction"
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "target_direction"
    ]

    if y_train.nunique() < 2:
        return None

    model = MODELS[
        model_name
    ](
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    return evaluate_classifier(
        y_test,
        predictions,
    )


def run_experiment(
    asset_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
):
    periods = get_walk_forward_periods(
        asset_df
    )

    rows = []

    for (
        test_date,
        train_df,
        test_df,
    ) in periods:

        metrics = evaluate_period(
            train_df,
            test_df,
            feature_columns,
            model_name,
        )

        if metrics is None:
            continue

        rows.append(
            {
                "test_date": test_date,
                "accuracy": metrics.accuracy,
                "balanced_accuracy": (
                    metrics.balanced_accuracy
                ),
                "precision": metrics.precision,
                "recall": metrics.recall,
                "samples": metrics.sample_count,
            }
        )

    if not rows:
        return None

    result = pd.DataFrame(rows)

    return {
        "periods": len(result),
        "mean_accuracy": (
            result["accuracy"].mean()
        ),
        "mean_balanced_accuracy": (
            result[
                "balanced_accuracy"
            ].mean()
        ),
        "median_balanced_accuracy": (
            result[
                "balanced_accuracy"
            ].median()
        ),
        "std_balanced_accuracy": (
            result[
                "balanced_accuracy"
            ].std()
        ),
        "mean_precision": (
            result["precision"].mean()
        ),
        "mean_recall": (
            result["recall"].mean()
        ),
    }


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

    summary_rows = []

    for asset, feature_sets in (
        FEATURE_SETS.items()
    ):

        asset_df = features[
            features["asset"] == asset
        ].copy()

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

        for (
            feature_set_name,
            feature_columns,
        ) in feature_sets.items():

            print(
                "\n"
                f"FEATURE SET: "
                f"{feature_set_name}"
            )

            for model_name in MODELS:

                print(
                    f"\nRunning: "
                    f"{model_name}"
                )

                result = run_experiment(
                    asset_df,
                    feature_columns,
                    model_name,
                )

                if result is None:
                    print(
                        "  No valid periods."
                    )
                    continue

                print(
                    "  Periods:",
                    result["periods"],
                )

                print(
                    "  Mean accuracy:",
                    f"{result['mean_accuracy']:.4f}",
                )

                print(
                    "  Mean balanced accuracy:",
                    f"{result['mean_balanced_accuracy']:.4f}",
                )

                print(
                    "  Median balanced accuracy:",
                    f"{result['median_balanced_accuracy']:.4f}",
                )

                print(
                    "  Std balanced accuracy:",
                    f"{result['std_balanced_accuracy']:.4f}",
                )

                print(
                    "  Mean precision:",
                    f"{result['mean_precision']:.4f}",
                )

                print(
                    "  Mean recall:",
                    f"{result['mean_recall']:.4f}",
                )

                summary_rows.append(
                    {
                        "asset": asset,
                        "feature_set": (
                            feature_set_name
                        ),
                        "model": model_name,
                        **result,
                    }
                )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "FINAL ASSET-LEVEL NEWS "
        "INCREMENTAL VALUE"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:.4f}"
            ),
        )
    )


if __name__ == "__main__":
    main()