from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.build_features import build_hourly_features
from src.modeling.dataset import build_modeling_dataset
from src.modeling.evaluation import evaluate_classifier
from src.modeling.logistic import fit_logistic_model
from src.modeling.splits import chronological_split


MARKET_PATH = "data/raw/market/research_market_15m.csv"
NEWS_PATH = "data/raw/news_sample.csv"
REDDIT_PATH = "data/raw/reddit_sample.csv"

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

    assets = sorted(
        features["asset"].unique()
    )

    print("=== PER-ASSET LOGISTIC REGRESSION ===")

    for asset in assets:
        print(f"\n=== {asset} ===")

        asset_features = features[
            features["asset"] == asset
        ].copy()

        split = chronological_split(
            asset_features,
            train_end=TRAIN_END,
            validation_end=VALIDATION_END,
        )

        train = build_modeling_dataset(
            split.train
        )

        validation = build_modeling_dataset(
            split.validation
        )

        test = build_modeling_dataset(
            split.test
        )

        print(
            "Train:",
            len(train.X),
            "Validation:",
            len(validation.X),
            "Test:",
            len(test.X),
        )

        print(
            "Train classes:",
            train.y.value_counts()
            .sort_index()
            .to_dict(),
        )

        model = fit_logistic_model(
            train.X,
            train.y,
        )

        validation_predictions = model.predict(
            validation.X
        )

        test_predictions = model.predict(
            test.X
        )

        validation_metrics = evaluate_classifier(
            validation.y,
            validation_predictions,
        )

        test_metrics = evaluate_classifier(
            test.y,
            test_predictions,
        )

        print("\nValidation")
        print(
            "  Accuracy:",
            validation_metrics.accuracy,
        )
        print(
            "  Balanced accuracy:",
            validation_metrics.balanced_accuracy,
        )
        print(
            "  Precision:",
            validation_metrics.precision,
        )
        print(
            "  Recall:",
            validation_metrics.recall,
        )
        print(
            "  Confusion matrix:",
            validation_metrics.confusion_matrix,
        )

        print("\nTest")
        print(
            "  Accuracy:",
            test_metrics.accuracy,
        )
        print(
            "  Balanced accuracy:",
            test_metrics.balanced_accuracy,
        )
        print(
            "  Precision:",
            test_metrics.precision,
        )
        print(
            "  Recall:",
            test_metrics.recall,
        )
        print(
            "  Confusion matrix:",
            test_metrics.confusion_matrix,
        )


if __name__ == "__main__":
    main()