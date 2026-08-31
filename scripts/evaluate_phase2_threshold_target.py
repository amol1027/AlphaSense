from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from src.features.market_features import (
    add_normalized_market_features,
)
from src.features.horizon_targets import (
    add_horizon_target,
)


INPUT_PATH = Path(
    "data/raw/market/phase1_research_market_15m.csv"
)

FINAL_TEST_START = pd.Timestamp(
    "2026-08-10",
    tz="UTC",
)

HORIZON = pd.Timedelta(hours=1)

THRESHOLD = 0.00203666

MARKET_FEATURES = [
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

LABELS = [
    "DOWN",
    "NEUTRAL",
    "UP",
]


def load_market_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    duplicate_count = df.duplicated(
        ["asset", "exchange", "timestamp"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            f"market observations"
        )

    return (
        df.sort_values(
            ["asset", "exchange", "timestamp"]
        )
        .reset_index(drop=True)
    )


def add_three_class_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    returns = result["target_return"]

    result["target_class"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    result.loc[
        returns < -THRESHOLD,
        "target_class",
    ] = "DOWN"

    result.loc[
        returns.abs() <= THRESHOLD,
        "target_class",
    ] = "NEUTRAL"

    result.loc[
        returns > THRESHOLD,
        "target_class",
    ] = "UP"

    return result


def prepare_dataset(
    market: pd.DataFrame,
) -> pd.DataFrame:

    features = add_normalized_market_features(
        market
    )

    features = add_horizon_target(
        features,
        HORIZON,
    )

    features = add_three_class_target(
        features
    )

    # Important:
    # sentiment features are not present in the
    # raw market source. They remain a separate
    # optional experiment and are checked explicitly.
    return features


def evaluate_baseline(
    y: pd.Series,
) -> dict:

    majority = y.mode().iloc[0]

    predictions = pd.Series(
        majority,
        index=y.index,
    )

    return {
        "model": "Majority baseline",
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y,
            predictions,
        ),
        "macro_f1": f1_score(
            y,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    }




def evaluate_logistic(
    df: pd.DataFrame,
    features: list[str],
) -> dict:

    usable = df[
        features + ["target_class"]
    ].dropna()

    if usable.empty:
        raise RuntimeError(
            "No usable observations for "
            "Logistic Regression."
        )

    X = usable[features]
    y = usable["target_class"]

    if y.nunique() < 3:
        raise RuntimeError(
            "Three target classes are not all present."
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_scaled,
        y,
    )

    predictions = model.predict(
        X_scaled
    )

    return {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y,
            predictions,
        ),
        "macro_f1": f1_score(
            y,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    }

def fit_predict_logistic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.Series, pd.Series]:
    train = train_df[
        features + ["target_class"]
    ].dropna()

    test = test_df[
        features + ["target_class"]
    ].dropna()

    if train.empty or test.empty:
        raise RuntimeError(
            "Insufficient train/test observations."
        )

    if train["target_class"].nunique() < 3:
        raise RuntimeError(
            "Training data does not contain all "
            "three target classes."
        )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[features]
    )

    X_test = scaler.transform(
        test[features]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_train,
        train["target_class"],
    )

    predictions = pd.Series(
        model.predict(X_test),
        index=test.index,
        dtype="string",
    )

    return (
        test["target_class"],
        predictions,
    )


def evaluate_predictions(
    y_true: pd.Series,
    predictions: pd.Series,
    model_name: str,
) -> dict:

    return {
        "model": model_name,
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    }



def print_class_distribution(
    y: pd.Series,
) -> None:

    counts = (
        y.value_counts()
        .reindex(LABELS)
        .fillna(0)
        .astype(int)
    )

    total = counts.sum()

    for label in LABELS:
        count = counts[label]

        print(
            f"{label:<8} "
            f"{count:>6,} "
            f"({count / total:.2%})"
        )


def main() -> None:

    print("=" * 80)
    print(
        "PHASE 2.2C — THREE-CLASS OUT-OF-SAMPLE EXPERIMENT"
    )
    print("=" * 80)

    market = load_market_data()

    print()
    print(f"Input: {INPUT_PATH}")
    print(f"Market rows: {len(market):,}")
    print(
        f"Horizon: {HORIZON}"
    )
    print(
        f"Frozen threshold: ±{THRESHOLD:.6%}"
    )
    print(
        f"Locked test starts: "
        f"{FINAL_TEST_START}"
    )

    data = prepare_dataset(
        market
    )

    # --------------------------------------------------------------
    # Chronological development split
    # --------------------------------------------------------------

    OOS_START = pd.Timestamp(
        "2026-07-01",
        tz="UTC",
    )

    train = data[
        data["timestamp"] < OOS_START
    ].copy()

    oos = data[
        (data["timestamp"] >= OOS_START)
        & (data["timestamp"] < FINAL_TEST_START)
    ].copy()

    locked = data[
        data["timestamp"] >= FINAL_TEST_START
    ].copy()

    train = train[
        train["target_class"].notna()
    ].copy()

    oos = oos[
        oos["target_class"].notna()
    ].copy()

    print()
    print("=" * 80)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 80)

    print(
        f"Training period: "
        f"< {OOS_START}"
    )

    print(
        f"OOS period: "
        f"{OOS_START} → "
        f"{FINAL_TEST_START}"
    )

    print(
        f"Locked period: "
        f">= {FINAL_TEST_START}"
    )

    print()
    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"OOS rows:      {len(oos):,}"
    )

    print(
        f"Locked rows:   {len(locked):,}"
    )

    results = []

    for asset in sorted(
        train["asset"].unique()
    ):

        train_asset = train[
            train["asset"] == asset
        ].copy()

        oos_asset = oos[
            oos["asset"] == asset
        ].copy()

        print()
        print("=" * 80)
        print(f"ASSET: {asset}")
        print("=" * 80)

        print()
        print("TRAINING TARGET DISTRIBUTION")
        print("-" * 80)

        print_class_distribution(
            train_asset["target_class"]
        )

        print()
        print("OOS TARGET DISTRIBUTION")
        print("-" * 80)

        print_class_distribution(
            oos_asset["target_class"]
        )

        # ----------------------------------------------------------
        # Majority baseline
        # ----------------------------------------------------------

        majority = (
            train_asset["target_class"]
            .mode()
            .iloc[0]
        )

        baseline_predictions = pd.Series(
            majority,
            index=oos_asset.index,
            dtype="string",
        )

        y_true = oos_asset[
            "target_class"
        ]

        baseline_result = evaluate_predictions(
            y_true,
            baseline_predictions,
            "Majority baseline",
        )

        print()
        print("BASELINE")
        print("-" * 80)

        print(
            pd.DataFrame(
                [baseline_result]
            ).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        results.append(
            {
                "asset": asset,
                **baseline_result,
            }
        )

        # ----------------------------------------------------------
        # Market Logistic
        # ----------------------------------------------------------

        y_true, predictions = (
            fit_predict_logistic(
                train_asset,
                oos_asset,
                MARKET_FEATURES,
            )
        )

        market_result = evaluate_predictions(
            y_true,
            predictions,
            "Market Logistic",
        )

        print()
        print("MARKET-ONLY LOGISTIC")
        print("-" * 80)

        print(
            pd.DataFrame(
                [market_result]
            ).to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        results.append(
            {
                "asset": asset,
                **market_result,
            }
        )

        # ----------------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------------

        print()
        print("CONFUSION MATRIX")
        print("-" * 80)

        matrix = confusion_matrix(
            y_true,
            predictions,
            labels=LABELS,
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                f"actual_{label}"
                for label in LABELS
            ],
            columns=[
                f"pred_{label}"
                for label in LABELS
            ],
        )

        print(
            matrix_df.to_string()
        )

        # ----------------------------------------------------------
        # Per-class recall
        # ----------------------------------------------------------

        recalls = (
            matrix.diagonal()
            / matrix.sum(axis=1)
        )

        print()
        print("PER-CLASS RECALL")
        print("-" * 80)

        for label, recall in zip(
            LABELS,
            recalls,
        ):
            print(
                f"{label:<8}: {recall:.4f}"
            )

    print()
    print("=" * 80)
    print("LOCKED HOLDOUT PROTECTION")
    print("=" * 80)

    print(
        f"Locked rows available in source: "
        f"{len(locked):,}"
    )

    print(
        "Locked rows were not used for:"
    )

    print(
        "  - threshold selection"
    )

    print(
        "  - model fitting"
    )

    print(
        "  - model selection"
    )

    print(
        "  - metric comparison"
    )

    print()
    print("=" * 80)
    print(
        "PHASE 2.2C OUT-OF-SAMPLE EXPERIMENT COMPLETE"
    )
    print("=" * 80)

if __name__ == "__main__":
    main()