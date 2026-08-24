from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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


def describe_probabilities(
    name: str,
    probabilities: np.ndarray,
) -> None:
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    print()
    print(name)
    print("-" * 70)

    print(
        "Min:",
        f"{probabilities.min():.6f}",
    )

    print(
        "25%:",
        f"{np.percentile(probabilities, 25):.6f}",
    )

    print(
        "Median:",
        f"{np.median(probabilities):.6f}",
    )

    print(
        "75%:",
        f"{np.percentile(probabilities, 75):.6f}",
    )

    print(
        "Max:",
        f"{probabilities.max():.6f}",
    )

    print(
        "Mean:",
        f"{probabilities.mean():.6f}",
    )

    print(
        "Std:",
        f"{probabilities.std():.6f}",
    )

    print(
        "Below 0.10:",
        int((probabilities < 0.10).sum()),
    )

    print(
        "0.10 - 0.25:",
        int(
            (
                (probabilities >= 0.10)
                & (probabilities < 0.25)
            ).sum()
        ),
    )

    print(
        "0.25 - 0.40:",
        int(
            (
                (probabilities >= 0.25)
                & (probabilities < 0.40)
            ).sum()
        ),
    )

    print(
        "0.40 - 0.50:",
        int(
            (
                (probabilities >= 0.40)
                & (probabilities < 0.50)
            ).sum()
        ),
    )

    print(
        "0.50 - 0.60:",
        int(
            (
                (probabilities >= 0.50)
                & (probabilities < 0.60)
            ).sum()
        ),
    )

    print(
        "0.60 - 0.75:",
        int(
            (
                (probabilities >= 0.60)
                & (probabilities < 0.75)
            ).sum()
        ),
    )

    print(
        "0.75 - 0.90:",
        int(
            (
                (probabilities >= 0.75)
                & (probabilities < 0.90)
            ).sum()
        ),
    )

    print(
        ">= 0.90:",
        int((probabilities >= 0.90).sum()),
    )


def main():
    print("=" * 70)
    print("PHASE 1 PROBABILITY DIAGNOSTIC")
    print("=" * 70)

    print()
    print("Input:", INPUT_PATH)
    print("Final test starts:", FINAL_TEST_START)

    df = pd.read_csv(INPUT_PATH)

    df["prediction_timestamp"] = pd.to_datetime(
        df["prediction_timestamp"],
        utc=True,
    )

    for asset, config in CANDIDATES.items():

        print()
        print("=" * 70)
        print(f"ASSET: {asset}")
        print("=" * 70)

        asset_df = df[
            df["asset"] == asset
        ].copy()

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

        classifier = model.classifier

        # --------------------------------------------------
        # Training probabilities
        # --------------------------------------------------

        train_probabilities = (
            classifier.predict_proba(
                (
                    model.scaler.transform(
                        X_train
                    )
                    if asset == "RELIANCE"
                    else X_train
                )
            )[:, 1]
        )

        # --------------------------------------------------
        # Test probabilities
        # --------------------------------------------------

        test_probabilities = (
            classifier.predict_proba(
                (
                    model.scaler.transform(
                        X_test
                    )
                    if asset == "RELIANCE"
                    else X_test
                )
            )[:, 1]
        )

        describe_probabilities(
            "TRAIN P(UP)",
            train_probabilities,
        )

        describe_probabilities(
            "TEST P(UP)",
            test_probabilities,
        )

        # --------------------------------------------------
        # Probability vs actual outcome
        # --------------------------------------------------

        diagnostic = pd.DataFrame(
            {
                "probability": test_probabilities,
                "actual": y_test.to_numpy(),
                "timestamp": test_df[
                    "prediction_timestamp"
                ].to_numpy(),
            }
        )

        print()
        print("TEST PROBABILITY BY ACTUAL CLASS")
        print("-" * 70)

        for actual_class in [0, 1]:

            values = diagnostic.loc[
                diagnostic["actual"]
                == actual_class,
                "probability",
            ]

            print(
                f"Actual {actual_class}: "
                f"n={len(values)}, "
                f"mean={values.mean():.6f}, "
                f"median={values.median():.6f}"
            )

        # --------------------------------------------------
        # Default threshold
        # --------------------------------------------------

        default_predictions = (
            test_probabilities >= 0.50
        ).astype(int)

        print()
        print("DEFAULT 0.50 THRESHOLD")
        print("-" * 70)

        print(
            "Predicted positives:",
            int(default_predictions.sum()),
        )

        print(
            "Predicted positive rate:",
            f"{default_predictions.mean():.4f}",
        )

        # --------------------------------------------------
        # Threshold diagnostic ONLY
        #
        # This does NOT change the frozen evaluation.
        # It simply shows how predictions would change.
        # --------------------------------------------------

        print()
        print("THRESHOLD SENSITIVITY")
        print("-" * 70)

        for threshold in [
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
        ]:

            predictions = (
                test_probabilities >= threshold
            ).astype(int)

            print(
                f"Threshold {threshold:.2f}: "
                f"{int(predictions.sum()):3d}/"
                f"{len(predictions)} positives "
                f"({predictions.mean():.3f})"
            )


if __name__ == "__main__":
    main()