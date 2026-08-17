from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.build_features import build_hourly_features
from src.modeling.dataset import build_train_validation_test
from src.modeling.evaluation import evaluate_classifier
from src.modeling.logistic import fit_logistic_model


MARKET_PATH = (
    "data/raw/market/research_market_15m.csv"
)

NEWS_PATH = (
    "data/raw/news_sample.csv"
)

REDDIT_PATH = (
    "data/raw/reddit_sample.csv"
)

TRAIN_END = pd.Timestamp(
    "2026-07-25",
    tz="UTC",
)

VALIDATION_END = pd.Timestamp(
    "2026-08-02",
    tz="UTC",
)


def main():
    features = build_hourly_features(
        MARKET_PATH,
        NEWS_PATH,
        REDDIT_PATH,
    )

    split = build_train_validation_test(
        features,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
    )

    model = fit_logistic_model(
        split.train.X,
        split.train.y,
    )

    predictions = model.predict(
        split.test.X
    )

    # Recover the test metadata directly from
    # the chronological split so asset labels
    # remain aligned with the predictions.
    from src.modeling.splits import chronological_split

    raw_split = chronological_split(
        features,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
    )

    test_metadata = raw_split.test[
    raw_split.test["target_direction"].notna()][["asset", "exchange"]].reset_index(drop=True)

    test_actual = (
    split.test.y.reset_index(drop=True)
    )

    test_predictions = (
        predictions.reset_index(drop=True)
    )
    assert len(test_metadata) == len(test_actual)
    assert len(test_actual) == len(test_predictions)

    results = pd.DataFrame(
        {
            "asset": test_metadata["asset"],
            "actual": test_actual,
            "prediction": test_predictions,
        }
    )

    print("=== LOGISTIC TEST RESULTS BY ASSET ===")

    for asset, group in results.groupby(
        "asset"
    ):
        metrics = evaluate_classifier(
            group["actual"],
            group["prediction"],
        )

        print(f"\n{asset}")
        print(
            "  Samples:",
            metrics.sample_count,
        )
        print(
            "  Accuracy:",
            metrics.accuracy,
        )
        print(
            "  Balanced accuracy:",
            metrics.balanced_accuracy,
        )
        print(
            "  Precision:",
            metrics.precision,
        )
        print(
            "  Recall:",
            metrics.recall,
        )
        print(
            "  Confusion matrix:",
            metrics.confusion_matrix,
        )


if __name__ == "__main__":
    main()