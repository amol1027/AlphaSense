from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.build_features import (
    build_hourly_features,
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


# Each test period is one trading day.
#
# The model uses all observations before that day
# for training and predicts that day's observations.
#
# We skip the earliest days so that the initial
# training set is reasonably large.
MIN_TRAIN_DAYS = 10


def evaluate_period(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
):
    train_df = train_df.dropna(
        subset=["target_direction"]
    ).reset_index(drop=True)

    test_df = test_df.dropna(
        subset=["target_direction"]
    ).reset_index(drop=True)

    if train_df.empty or test_df.empty:
        return None

    X_train = train_df[
        feature_columns
    ].copy()

    y_train = train_df[
        "target_direction"
    ].copy()

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        "target_direction"
    ].copy()

    model = fit_logistic_model(
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

    return metrics


def run_walk_forward(
    features: pd.DataFrame,
    feature_columns: list[str],
    asset: str | None = None,
):
    if asset is not None:
        data = features[
            features["asset"] == asset
        ].copy()
    else:
        data = features.copy()

    data = data.sort_values(
        "prediction_timestamp"
    ).reset_index(drop=True)

    data["date"] = (
        data["prediction_timestamp"]
        .dt.date
    )

    trading_dates = sorted(
        data["date"].unique()
    )

    results = []

    for index in range(
        MIN_TRAIN_DAYS,
        len(trading_dates),
    ):
        test_date = trading_dates[index]

        train_dates = trading_dates[
            :index
        ]

        train_df = data[
            data["date"].isin(
                train_dates
            )
        ].copy()

        test_df = data[
            data["date"] == test_date
        ].copy()

        metrics = evaluate_period(
            train_df,
            test_df,
            feature_columns,
        )

        if metrics is None:
            continue

        results.append(
            {
                "asset": (
                    asset
                    if asset is not None
                    else "ALL"
                ),
                "test_date": test_date,
                "train_samples": len(
                    train_df.dropna(
                        subset=[
                            "target_direction"
                        ]
                    )
                ),
                "test_samples": (
                    metrics.sample_count
                ),
                "accuracy": (
                    metrics.accuracy
                ),
                "balanced_accuracy": (
                    metrics.balanced_accuracy
                ),
                "precision": (
                    metrics.precision
                ),
                "recall": (
                    metrics.recall
                ),
            }
        )

    return pd.DataFrame(
        results
    )


def summarize_results(
    results: pd.DataFrame,
):
    if results.empty:
        return pd.Series(
            dtype=float
        )

    return pd.Series(
        {
            "periods": len(results),
            "mean_accuracy": (
                results["accuracy"]
                .mean()
            ),
            "mean_balanced_accuracy": (
                results[
                    "balanced_accuracy"
                ].mean()
            ),
            "median_balanced_accuracy": (
                results[
                    "balanced_accuracy"
                ].median()
            ),
            "std_balanced_accuracy": (
                results[
                    "balanced_accuracy"
                ].std()
            ),
            "mean_precision": (
                results["precision"]
                .mean()
            ),
            "mean_recall": (
                results["recall"]
                .mean()
            ),
        }
    )


def compare_news_to_market(
    features: pd.DataFrame,
    asset: str | None = None,
):
    market = run_walk_forward(
        features,
        MARKET_FEATURES,
        asset=asset,
    )

    news = run_walk_forward(
        features,
        MARKET_FEATURES + NEWS_FEATURES,
        asset=asset,
    )

    comparison = market.merge(
        news,
        on=["test_date"],
        suffixes=(
            "_market",
            "_news",
        ),
    )

    comparison["balanced_accuracy_delta"] = (
        comparison[
            "balanced_accuracy_news"
        ]
        - comparison[
            "balanced_accuracy_market"
        ]
    )

    return comparison[
        [
            "test_date",
            "balanced_accuracy_market",
            "balanced_accuracy_news",
            "balanced_accuracy_delta",
        ]
    ]


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

    print(
        "Trading days:",
        features[
            "prediction_timestamp"
        ].dt.date.nunique(),
    )

    all_summary_rows = []

    assets = [
        None,
        "RELIANCE",
        "TCS",
    ]

    for asset in assets:

        asset_label = (
            asset
            if asset is not None
            else "ALL ASSETS"
        )

        print(
            "\n"
            + "=" * 70
        )
        print(
            f"WALK-FORWARD: {asset_label}"
        )
        print(
            "=" * 70
        )

        for (
            experiment,
            feature_columns,
        ) in EXPERIMENTS.items():

            print(
                f"\nRunning: {experiment}"
            )

            results = run_walk_forward(
                features,
                feature_columns,
                asset=asset,
            )

            summary = summarize_results(
                results
            )

            print(
                f"  Periods: "
                f"{int(summary['periods'])}"
            )

            print(
                "  Mean accuracy:",
                f"{summary['mean_accuracy']:.4f}",
            )

            print(
                "  Mean balanced accuracy:",
                f"{summary['mean_balanced_accuracy']:.4f}",
            )

            print(
                "  Median balanced accuracy:",
                f"{summary['median_balanced_accuracy']:.4f}",
            )

            print(
                "  Std balanced accuracy:",
                f"{summary['std_balanced_accuracy']:.4f}",
            )

            print(
                "  Mean precision:",
                f"{summary['mean_precision']:.4f}",
            )

            print(
                "  Mean recall:",
                f"{summary['mean_recall']:.4f}",
            )

            all_summary_rows.append(
                {
                    "asset": asset_label,
                    "experiment": experiment,
                    **summary.to_dict(),
                }
            )

    summary_df = pd.DataFrame(
        all_summary_rows
    )

    print(
        "\n\n"
        + "=" * 70
    )
    print(
        "NEWS VS MARKET-ONLY BY TEST PERIOD"
    )
    print(
        "=" * 70
    )

    for asset in [
        None,
        "RELIANCE",
        "TCS",
    ]:

        label = (
            asset
            if asset is not None
            else "ALL ASSETS"
        )

        comparison = compare_news_to_market(
            features,
            asset=asset,
        )

        print(
            f"\n{label}"
        )

        print(
            comparison.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        wins = (
            comparison[
                "balanced_accuracy_delta"
            ] > 0
        ).sum()

        losses = (
            comparison[
                "balanced_accuracy_delta"
            ] < 0
        ).sum()

        ties = (
            comparison[
                "balanced_accuracy_delta"
            ] == 0
        ).sum()

        print(
            f"\nNews wins: {wins}"
            f" | losses: {losses}"
            f" | ties: {ties}"
        )

    print(
        "\n\n"
        + "=" * 70
    )
    print(
        "FINAL WALK-FORWARD SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()