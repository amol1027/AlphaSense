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

REDUCED_ENGINEERED_MARKET_FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "volume_change",
]


FEATURE_SETS = {
    "Market only": MARKET_FEATURES,

    "Engineered Market": (
        ENGINEERED_MARKET_FEATURES
    ),

    "Reduced Engineered Market": (
        REDUCED_ENGINEERED_MARKET_FEATURES
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
# WALK-FORWARD PERIODS
# ==========================================================

def get_walk_forward_periods(
    features: pd.DataFrame,
    asset: str | None = None,
):
    """
    Build expanding-window walk-forward periods.

    The first MIN_TRAIN_DAYS are used for the initial
    training window. Each subsequent trading day becomes
    one test period.
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
# DATA PREPARATION
# ==========================================================

def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
):
    """
    Remove rows with missing targets or selected
    features.

    Only the historical training data is used to
    fit the model.
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

    if train_df.empty:
        return None, None

    if test_df.empty:
        return None, None

    return train_df, test_df


# ==========================================================
# ONE PERIOD
# ==========================================================

def evaluate_period(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
):
    """
    Fit one model using only the training portion and
    evaluate it on the following test period.
    """

    train_df, test_df = prepare_data(
        train_df,
        test_df,
        feature_columns,
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

    if y_train.nunique() < 2:
        return None

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


# ==========================================================
# WALK-FORWARD MODEL
# ==========================================================

def run_walk_forward(
    features: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    asset: str | None = None,
):
    """
    Run one model over all walk-forward periods.
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

        metrics = evaluate_period(
            train_df,
            test_df,
            feature_columns,
            model_name,
        )

        if metrics is None:
            continue

        results.append(
            {
                "asset": (
                    asset
                    if asset is not None
                    else "ALL ASSETS"
                ),
                "test_date": test_date,
                "model": model_name,
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
                "samples": (
                    metrics.sample_count
                ),
            }
        )

    return pd.DataFrame(
        results
    )


# ==========================================================
# SUMMARY
# ==========================================================

def summarize(
    results: pd.DataFrame,
):
    """
    Aggregate metrics across walk-forward periods.
    """

    if results.empty:
        return None

    return {
        "periods": len(results),
        "mean_accuracy": (
            results["accuracy"].mean()
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
            results["precision"].mean()
        ),
        "mean_recall": (
            results["recall"].mean()
        ),
    }


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
        "\n"
        + "=" * 70
    )

    print(
        "ENGINEERED MARKET FEATURE "
        "WALK-FORWARD COMPARISON"
    )

    print(
        "=" * 70
    )

    summary_rows = []

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
            f"WALK-FORWARD: "
            f"{asset_label}"
        )

        print(
            "=" * 70
        )

        for (
            feature_set_name,
            feature_columns,
        ) in FEATURE_SETS.items():

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

                results = run_walk_forward(
                    features,
                    feature_columns,
                    model_name,
                    asset=asset,
                )

                summary = summarize(
                    results
                )

                if summary is None:
                    print(
                        "  No valid periods."
                    )
                    continue

                print(
                    "  Periods:",
                    summary["periods"],
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

                summary_rows.append(
                    {
                        "asset": asset_label,
                        "feature_set": (
                            feature_set_name
                        ),
                        "model": model_name,
                        **summary,
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
        "FINAL ENGINEERED FEATURE "
        "COMPARISON"
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