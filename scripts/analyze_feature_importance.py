from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features.build_features import (
    build_hourly_features,
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


FEATURE_SETS = {
    "Market only": MARKET_FEATURES,

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


def get_walk_forward_periods(
    features: pd.DataFrame,
    asset: str | None = None,
):
    """
    Return the same expanding-window periods used
    by evaluate_walk_forward.py.
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


def prepare_training_data(
    train_df: pd.DataFrame,
    feature_columns: list[str],
):
    """
    Prepare only the historical training observations.
    """

    required = [
        "target_direction"
    ] + feature_columns

    train_df = train_df.dropna(
        subset=required
    ).copy()

    if train_df.empty:
        return None, None

    X_train = train_df[
        feature_columns
    ].copy()

    y_train = train_df[
        "target_direction"
    ].copy()

    return X_train, y_train


def extract_importance(
    model_name: str,
    model,
    feature_columns: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Extract feature importance from a fitted model.

    Logistic Regression:
        absolute standardized coefficient magnitude
        plus coefficient direction.

    Random Forest:
        sklearn impurity-based feature importance.

    HistGradientBoosting:
        permutation importance calculated only on
        the historical training data.

    No future/test observations are used.
    """

    if model_name == "Logistic Regression":

        coefficients = (
            model.classifier.coef_[0]
        )

        rows = []

        for feature, coefficient in zip(
            feature_columns,
            coefficients,
        ):
            rows.append(
                {
                    "feature": feature,
                    "importance": abs(
                        float(coefficient)
                    ),
                    "signed_importance": (
                        float(coefficient)
                    ),
                    "direction": (
                        "positive"
                        if coefficient > 0
                        else (
                            "negative"
                            if coefficient < 0
                            else "zero"
                        )
                    ),
                }
            )

        return rows

    classifier = model.classifier

    if model_name == "Random Forest":

        importances = (
            classifier.feature_importances_
        )

        return [
            {
                "feature": feature,
                "importance": float(
                    importance
                ),
                "signed_importance": np.nan,
                "direction": "n/a",
            }
            for feature, importance in zip(
                feature_columns,
                importances,
            )
        ]

    if model_name == "HistGradientBoosting":

        from sklearn.inspection import (
            permutation_importance,
        )

        permutation = (
            permutation_importance(
                classifier,
                X_train,
                y_train,
                n_repeats=5,
                random_state=42,
                scoring="balanced_accuracy",
                n_jobs=-1,
            )
        )

        importances = (
            permutation.importances_mean
        )

        return [
            {
                "feature": feature,
                "importance": float(
                    importance
                ),
                "signed_importance": np.nan,
                "direction": "n/a",
            }
            for feature, importance in zip(
                feature_columns,
                importances,
            )
        ]

    raise ValueError(
        "Unsupported model for feature "
        f"importance analysis: {model_name}"
    )

def run_importance_analysis(
    features: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    asset: str | None = None,
):
    """
    Fit the model separately on every historical
    walk-forward training period and collect
    feature importance.

    No future test data is used.
    """

    periods = get_walk_forward_periods(
        features,
        asset=asset,
    )

    rows = []

    for period_index, (
        test_date,
        train_df,
        _test_df,
    ) in enumerate(periods, start=1):

        X_train, y_train = (
            prepare_training_data(
                train_df,
                feature_columns,
            )
        )

        if X_train is None:
            continue

        if y_train.nunique() < 2:
            continue

        fit_function = MODELS[
            model_name
        ]

        model = fit_function(
            X_train,
            y_train,
        )

        importances = extract_importance(
        model_name,
        model,
        feature_columns,
        X_train,
        y_train,)

        for row in importances:

            rows.append(
                {
                    "asset": (
                        asset
                        if asset is not None
                        else "ALL"
                    ),
                    "period": period_index,
                    "test_date": test_date,
                    "model": model_name,
                    "feature": row[
                        "feature"
                    ],
                    "importance": row[
                        "importance"
                    ],
                    "signed_importance": row[
                        "signed_importance"
                    ],
                    "direction": row[
                        "direction"
                    ],
                }
            )

    return pd.DataFrame(rows)


def summarize_importance(
    importance_df: pd.DataFrame,
):
    """
    Aggregate feature importance across
    walk-forward training periods.
    """

    if importance_df.empty:
        return pd.DataFrame()

    grouped = (
        importance_df
        .groupby(
            [
                "asset",
                "model",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            mean_importance=(
                "importance",
                "mean",
            ),
            median_importance=(
                "importance",
                "median",
            ),
            std_importance=(
                "importance",
                "std",
            ),
            periods=(
                "importance",
                "count",
            ),
        )
    )

    # Count periods in which the feature
    # had non-zero importance.
    nonzero = (
        importance_df
        .assign(
            nonzero=lambda df:
                df["importance"] > 0
        )
        .groupby(
            [
                "asset",
                "model",
                "feature",
            ]
        )["nonzero"]
        .sum()
        .reset_index(
            name="nonzero_periods"
        )
    )

    grouped = grouped.merge(
        nonzero,
        on=[
            "asset",
            "model",
            "feature",
        ],
        how="left",
    )

    # Logistic-only directional stability.
    signed = (
        importance_df[
            importance_df[
                "signed_importance"
            ].notna()
        ]
        .groupby(
            [
                "asset",
                "model",
                "feature",
            ]
        )["signed_importance"]
        .agg(
            mean_signed_importance="mean",
            positive_periods=lambda x: (
                x > 0
            ).sum(),
            negative_periods=lambda x: (
                x < 0
            ).sum(),
        )
        .reset_index()
    )

    grouped = grouped.merge(
        signed,
        on=[
            "asset",
            "model",
            "feature",
        ],
        how="left",
    )

    grouped[
        "importance_rank"
    ] = (
        grouped
        .groupby(
            [
                "asset",
                "model",
            ]
        )["mean_importance"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    return grouped.sort_values(
        [
            "asset",
            "model",
            "importance_rank",
        ]
    ).reset_index(
        drop=True
    )


def print_results(
    summary: pd.DataFrame,
    feature_set_name: str,
):
    print(
        "\n"
        + "=" * 70
    )

    print(
        f"FEATURE IMPORTANCE: "
        f"{feature_set_name}"
    )

    print(
        "=" * 70
    )

    for model_name in MODELS:

        model_summary = summary[
            summary["model"]
            == model_name
        ]

        if model_summary.empty:
            continue

        print(
            f"\n--- {model_name} ---"
        )

        columns = [
            "asset",
            "feature",
            "mean_importance",
            "median_importance",
            "std_importance",
            "nonzero_periods",
        ]

        if (
            model_name
            == "Logistic Regression"
        ):
            columns.extend(
                [
                    "mean_signed_importance",
                    "positive_periods",
                    "negative_periods",
                ]
            )

        print(
            model_summary[
                columns
            ].to_string(
                index=False,
                float_format=(
                    lambda x:
                    f"{x:.6f}"
                ),
            )
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

    all_assets = [
        None,
        "RELIANCE",
        "TCS",
    ]

    for (
        feature_set_name,
        feature_columns,
    ) in FEATURE_SETS.items():

        all_results = []

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"ANALYZING: "
            f"{feature_set_name}"
        )

        print(
            "=" * 70
        )

        for asset in all_assets:

            for model_name in MODELS:

                print(
                    f"  {asset or 'ALL'} | "
                    f"{model_name}"
                )

                result = (
                    run_importance_analysis(
                        features,
                        feature_columns,
                        model_name,
                        asset=asset,
                    )
                )

                if not result.empty:
                    all_results.append(
                        result
                    )

        if not all_results:
            print(
                "No importance results."
            )
            continue

        importance_df = pd.concat(
            all_results,
            ignore_index=True,
        )

        summary = (
            summarize_importance(
                importance_df
            )
        )

        print_results(
            summary,
            feature_set_name,
        )


if __name__ == "__main__":
    main()