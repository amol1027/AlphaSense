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


# ==========================================================
# FEATURE SETS
# ==========================================================

MARKET_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


ENGINEERED_MARKET_FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
]


NEWS_FEATURES = [
    "sentiment_mean",
    "sentiment_std",
    "news_count",
    "positive_ratio",
    "negative_ratio",
]


MODEL_FEATURE_SETS = {
    "Market only": (
        MARKET_FEATURES
    ),

    "Engineered Market": (
        ENGINEERED_MARKET_FEATURES
    ),

    "Market + News": (
        MARKET_FEATURES
        + NEWS_FEATURES
    ),

    "Engineered Market + News": (
        ENGINEERED_MARKET_FEATURES
        + NEWS_FEATURES
    ),
}


MODELS = {
    "Logistic Regression": fit_logistic_model,
    "Random Forest": fit_random_forest,
    "HistGradientBoosting": (
        fit_hist_gradient_boosting
    ),
}


MIN_TRAIN_DAYS = 10


# ==========================================================
# DATA PREPARATION
# ==========================================================

def prepare_period_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
):
    """
    Prepare one walk-forward period.

    Rows without a target or selected feature values
    are removed.

    Filtering is performed independently for each
    period so future information cannot influence
    training.
    """

    required_columns = [
        "target_direction"
    ] + feature_columns

    train_df = train_df.dropna(
        subset=required_columns
    ).copy()

    test_df = test_df.dropna(
        subset=required_columns
    ).copy()

    train_df = train_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    if train_df.empty or test_df.empty:
        return None, None

    return train_df, test_df


# ==========================================================
# MODEL EVALUATION
# ==========================================================

def evaluate_model_period(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
):
    """
    Fit one model using only the historical training
    period and evaluate it on the following test period.
    """

    train_df, test_df = (
        prepare_period_data(
            train_df,
            test_df,
            feature_columns,
        )
    )

    if train_df is None:
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

    fit_function = MODELS[
        model_name
    ]

    model = fit_function(
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


def evaluate_majority_period(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Evaluate a majority-class baseline.

    The majority class is calculated exclusively
    from historical training observations.
    """

    train_df = train_df.dropna(
        subset=["target_direction"]
    ).copy()

    test_df = test_df.dropna(
        subset=["target_direction"]
    ).copy()

    if train_df.empty or test_df.empty:
        return None

    y_train = train_df[
        "target_direction"
    ].astype(int)

    y_test = test_df[
        "target_direction"
    ].astype(int)

    majority_class = int(
        y_train
        .value_counts()
        .idxmax()
    )

    predictions = pd.Series(
        majority_class,
        index=y_test.index,
        dtype=int,
    )

    return evaluate_classifier(
        y_test,
        predictions,
    )


# ==========================================================
# WALK-FORWARD PERIODS
# ==========================================================

def get_walk_forward_periods(
    features: pd.DataFrame,
    asset: str | None = None,
):
    """
    Construct expanding-window chronological periods.

    Each test period represents one trading day.
    """

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
            data["date"].isin(
                train_dates
            )
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


# ==========================================================
# WALK-FORWARD MODEL RUNNERS
# ==========================================================

def run_model_walk_forward(
    features: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    asset: str | None = None,
):
    """
    Run one model through all expanding-window
    walk-forward periods.
    """

    periods = get_walk_forward_periods(
        features,
        asset=asset,
    )

    results = []

    for (
        test_date,
        train_df,
        test_df,
    ) in periods:

        metrics = evaluate_model_period(
            train_df,
            test_df,
            feature_columns,
            model_name,
        )

        if metrics is None:
            continue

        train_used, test_used = (
            prepare_period_data(
                train_df,
                test_df,
                feature_columns,
            )
        )

        results.append(
            {
                "asset": (
                    asset
                    if asset is not None
                    else "ALL"
                ),
                "test_date": test_date,
                "model": model_name,
                "train_samples": len(
                    train_used
                ),
                "test_samples": len(
                    test_used
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


def run_majority_walk_forward(
    features: pd.DataFrame,
    asset: str | None = None,
):
    """
    Run majority baseline over the exact same
    chronological periods.
    """

    periods = get_walk_forward_periods(
        features,
        asset=asset,
    )

    results = []

    for (
        test_date,
        train_df,
        test_df,
    ) in periods:

        metrics = evaluate_majority_period(
            train_df,
            test_df,
        )

        if metrics is None:
            continue

        train_clean = train_df.dropna(
            subset=["target_direction"]
        )

        majority_class = int(
            train_clean[
                "target_direction"
            ]
            .astype(int)
            .value_counts()
            .idxmax()
        )

        results.append(
            {
                "asset": (
                    asset
                    if asset is not None
                    else "ALL"
                ),
                "test_date": test_date,
                "model": "Majority baseline",
                "majority_class": (
                    majority_class
                ),
                "train_samples": len(
                    train_clean
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


# ==========================================================
# SUMMARY
# ==========================================================

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


def print_summary(
    summary: pd.Series,
):
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


# ==========================================================
# MAIN
# ==========================================================

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

    print(
        "\nEngineered market features:"
    )

    print(
        features[
            ENGINEERED_MARKET_FEATURES
        ]
        .describe()
        .to_string()
    )

    all_summary_rows = []

    assets = [
        None,
        "RELIANCE",
        "TCS",
    ]

    # ==================================================
    # Walk-forward comparison
    # ==================================================

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

        # ----------------------------------------------
        # Majority baseline
        # ----------------------------------------------

        print(
            "\n"
            "Running: Majority baseline"
        )

        majority_results = (
            run_majority_walk_forward(
                features,
                asset=asset,
            )
        )

        majority_summary = (
            summarize_results(
                majority_results
            )
        )

        print_summary(
            majority_summary
        )

        all_summary_rows.append(
            {
                "asset": asset_label,
                "feature_set": (
                    "Baseline"
                ),
                "model": (
                    "Majority baseline"
                ),
                **majority_summary.to_dict(),
            }
        )

        # ----------------------------------------------
        # Model comparison
        # ----------------------------------------------

        for (
            feature_set_name,
            feature_columns,
        ) in MODEL_FEATURE_SETS.items():

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

                results = (
                    run_model_walk_forward(
                        features,
                        feature_columns,
                        model_name,
                        asset=asset,
                    )
                )

                summary = summarize_results(
                    results
                )

                print_summary(
                    summary
                )

                all_summary_rows.append(
                    {
                        "asset": asset_label,
                        "feature_set": (
                            feature_set_name
                        ),
                        "model": model_name,
                        **summary.to_dict(),
                    }
                )

    # ==================================================
    # Final summary
    # ==================================================

    summary_df = pd.DataFrame(
        all_summary_rows
    )

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "FINAL MODEL COMPARISON"
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