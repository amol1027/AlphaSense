from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.modeling.logistic import fit_logistic_model
from src.modeling.tree_models import fit_hist_gradient_boosting


INPUT_PATH = "data/processed/phase1_features.csv"

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
        dtype=int,
    )


def main():
    print("=" * 70)
    print("PHASE 1 PREDICTION DISTRIBUTION DIAGNOSTIC")
    print("=" * 70)

    print()
    print("Input:", INPUT_PATH)
    print("Final test starts:", FINAL_TEST_START)

    df = pd.read_csv(INPUT_PATH)

    df["prediction_timestamp"] = pd.to_datetime(
        df["prediction_timestamp"],
        utc=True,
    )

    print()
    print("Feature rows:", len(df))

    results = []

    for asset, config in CANDIDATES.items():

        print()
        print("=" * 70)
        print(f"ASSET: {asset}")
        print("=" * 70)

        asset_df = df[
            df["asset"] == asset
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

        model = config["fit"](
            X_train,
            y_train,
        )

        predictions = pd.Series(
            model.predict(X_test),
            index=test_df.index,
            dtype=int,
        )

        baseline_predictions = majority_predict(
            y_train,
            y_test,
        )

        print()
        print(
            "Frozen candidate:",
            config["name"],
        )

        print()
        print("TRAIN")
        print("-" * 70)

        print(
            "Samples:",
            len(y_train),
        )

        print(
            "Actual class distribution:"
        )

        print(
            y_train.value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "Positive rate:",
            f"{y_train.mean():.4f}",
        )

        print()
        print("TEST ACTUAL")
        print("-" * 70)

        print(
            "Samples:",
            len(y_test),
        )

        print(
            "Actual class distribution:"
        )

        print(
            y_test.value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "Positive rate:",
            f"{y_test.mean():.4f}",
        )

        print()
        print("CANDIDATE PREDICTIONS")
        print("-" * 70)

        print(
            "Prediction distribution:"
        )

        print(
            predictions.value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "Positive prediction rate:",
            f"{predictions.mean():.4f}",
        )

        print()
        print("MAJORITY BASELINE")
        print("-" * 70)

        print(
            "Prediction distribution:"
        )

        print(
            baseline_predictions.value_counts()
            .sort_index()
            .to_string()
        )

        print(
            "Positive prediction rate:",
            f"{baseline_predictions.mean():.4f}",
        )

        # --------------------------------------------------
        # Confusion matrix manually
        # --------------------------------------------------

        tn = int(
            (
                (y_test == 0)
                & (predictions == 0)
            ).sum()
        )

        fp = int(
            (
                (y_test == 0)
                & (predictions == 1)
            ).sum()
        )

        fn = int(
            (
                (y_test == 1)
                & (predictions == 0)
            ).sum()
        )

        tp = int(
            (
                (y_test == 1)
                & (predictions == 1)
            ).sum()
        )

        print()
        print("CONFUSION MATRIX")
        print("-" * 70)

        print(
            "TN:",
            tn,
            " FP:",
            fp,
        )

        print(
            "FN:",
            fn,
            " TP:",
            tp,
        )

        # --------------------------------------------------
        # Daily prediction behavior
        # --------------------------------------------------

        diagnostic_df = test_df[
            [
                "prediction_timestamp",
                "target_direction",
            ]
        ].copy()

        diagnostic_df["prediction"] = (
            predictions
        )

        diagnostic_df["date"] = (
            diagnostic_df[
                "prediction_timestamp"
            ].dt.date
        )

        daily = (
            diagnostic_df
            .groupby("date")
            .agg(
                samples=(
                    "target_direction",
                    "count",
                ),
                actual_positive_rate=(
                    "target_direction",
                    "mean",
                ),
                predicted_positive_rate=(
                    "prediction",
                    "mean",
                ),
                predicted_zeros=(
                    "prediction",
                    lambda x: int(
                        (x == 0).sum()
                    ),
                ),
                predicted_ones=(
                    "prediction",
                    lambda x: int(
                        (x == 1).sum()
                    ),
                ),
            )
            .reset_index()
        )

        print()
        print("DAILY PREDICTION DISTRIBUTION")
        print("-" * 70)

        print(
            daily.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.4f}",
            )
        )

        results.append(
            {
                "asset": asset,
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "actual_positive_rate": (
                    y_test.mean()
                ),
                "predicted_positive_rate": (
                    predictions.mean()
                ),
                "predicted_zeros": int(
                    (predictions == 0).sum()
                ),
                "predicted_ones": int(
                    (predictions == 1).sum()
                ),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = pd.DataFrame(results)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()