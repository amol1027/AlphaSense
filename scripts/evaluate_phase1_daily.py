from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.modeling.evaluation import evaluate_classifier
from src.modeling.logistic import fit_logistic_model
from src.modeling.tree_models import fit_hist_gradient_boosting


FEATURE_PATH = (
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


def main():
    print("=" * 70)
    print("PHASE 1 DAILY LOCKED EVALUATION")
    print("=" * 70)

    print(
        f"Input: {FEATURE_PATH}"
    )

    df = pd.read_csv(
        FEATURE_PATH
    )

    df["prediction_timestamp"] = (
        pd.to_datetime(
            df["prediction_timestamp"],
            utc=True,
        )
    )

    print()
    print(
        f"Feature rows: {len(df):,}"
    )

    print()
    print("Final test starts:")
    print(FINAL_TEST_START)

    test = df[
        df["prediction_timestamp"]
        >= FINAL_TEST_START
    ].copy()

    print()
    print(
        f"Raw final-test rows: "
        f"{len(test):,}"
    )

    results = []

    for asset, config in CANDIDATES.items():

        print()
        print("=" * 70)
        print(f"ASSET: {asset}")
        print("=" * 70)

        asset_df = test[
            test["asset"] == asset
        ].copy()

        required = [
            "target_direction",
            *config["features"],
        ]

        asset_df = asset_df.dropna(
            subset=required
        )

        asset_df = asset_df.sort_values(
            "prediction_timestamp"
        )

        print(
            f"Usable test rows: "
            f"{len(asset_df):,}"
        )

        if asset_df.empty:
            print(
                "WARNING: No usable test rows."
            )
            continue

        train = df[
            (df["asset"] == asset)
            & (
                df["prediction_timestamp"]
                < FINAL_TEST_START
            )
        ].copy()

        train = train.dropna(
            subset=required
        )

        X_train = train[
            config["features"]
        ]

        y_train = train[
            "target_direction"
        ]

        model = config["fit"](
            X_train,
            y_train,
        )

        print(
            "Frozen candidate:",
            config["name"],
        )

        for prediction_date, day_df in (
            asset_df.groupby(
                asset_df[
                    "prediction_timestamp"
                ].dt.date
            )
        ):

            X_day = day_df[
                config["features"]
            ]

            y_day = day_df[
                "target_direction"
            ]

            predictions = model.predict(
                X_day
            )

            metrics = evaluate_classifier(
                y_day,
                predictions,
            )

            baseline_predictions = (
                majority_predict(
                    y_train,
                    y_day,
                )
            )

            baseline_metrics = (
                evaluate_classifier(
                    y_day,
                    baseline_predictions,
                )
            )

            print()
            print(
                f"Date: {prediction_date}"
            )

            print(
                f"  Samples: "
                f"{metrics.sample_count}"
            )

            print(
                f"  Candidate accuracy: "
                f"{metrics.accuracy:.4f}"
            )

            print(
                f"  Candidate balanced accuracy: "
                f"{metrics.balanced_accuracy:.4f}"
            )

            print(
                f"  Baseline accuracy: "
                f"{baseline_metrics.accuracy:.4f}"
            )

            print(
                f"  Baseline balanced accuracy: "
                f"{baseline_metrics.balanced_accuracy:.4f}"
            )

            print(
                f"  Balanced accuracy delta: "
                f"{(
                    metrics.balanced_accuracy
                    - baseline_metrics.balanced_accuracy
                ):.4f}"
            )

            results.append(
                {
                    "asset": asset,
                    "date": prediction_date,
                    "samples": metrics.sample_count,
                    "candidate_accuracy": (
                        metrics.accuracy
                    ),
                    "candidate_balanced_accuracy": (
                        metrics.balanced_accuracy
                    ),
                    "baseline_accuracy": (
                        baseline_metrics.accuracy
                    ),
                    "baseline_balanced_accuracy": (
                        baseline_metrics.balanced_accuracy
                    ),
                    "balanced_accuracy_delta": (
                        metrics.balanced_accuracy
                        - baseline_metrics.balanced_accuracy
                    ),
                }
            )

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("DAILY RESULTS SUMMARY")
    print("=" * 70)

    if results_df.empty:
        print("No results.")
        return

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 70)
    print("AVERAGE DAILY PERFORMANCE")
    print("=" * 70)

    summary = (
        results_df
        .groupby("asset")
        .agg(
            days=("date", "count"),
            total_samples=("samples", "sum"),
            mean_candidate_accuracy=(
                "candidate_accuracy",
                "mean",
            ),
            mean_candidate_balanced_accuracy=(
                "candidate_balanced_accuracy",
                "mean",
            ),
            mean_baseline_accuracy=(
                "baseline_accuracy",
                "mean",
            ),
            mean_baseline_balanced_accuracy=(
                "baseline_balanced_accuracy",
                "mean",
            ),
            mean_balanced_accuracy_delta=(
                "balanced_accuracy_delta",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 70)
    print("POSITIVE DAYS")
    print("=" * 70)

    positive_days = (
        results_df
        .assign(
            candidate_beats_baseline=(
                results_df[
                    "balanced_accuracy_delta"
                ] > 0
            )
        )
        .groupby("asset")
        .agg(
            positive_days=(
                "candidate_beats_baseline",
                "sum",
            ),
            total_days=(
                "candidate_beats_baseline",
                "count",
            ),
        )
        .reset_index()
    )

    print(
        positive_days.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()