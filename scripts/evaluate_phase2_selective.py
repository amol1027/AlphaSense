from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
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
    "2026-08-10 00:00:00",
    tz="UTC",
)

TRAIN_END = pd.Timestamp(
    "2026-07-01 00:00:00",
    tz="UTC",
)

HORIZON = pd.Timedelta(hours=1)

THRESHOLD = 0.00203666

PROBABILITY_PERCENTILE = 90

MARKET_FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
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


def prepare_dataset(
    market: pd.DataFrame,
) -> pd.DataFrame:

    result = add_normalized_market_features(
        market
    )

    result = add_horizon_target(
        result,
        HORIZON,
    )

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


def fit_model(
    train: pd.DataFrame,
) -> tuple[
    StandardScaler,
    LogisticRegression,
]:

    usable = train[
        MARKET_FEATURES + ["target_class"]
    ].dropna()

    if usable.empty:
        raise RuntimeError(
            "No usable training observations."
        )

    y = usable["target_class"]

    if y.nunique() < 3:
        raise RuntimeError(
            "Training data does not contain "
            "all three target classes."
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        usable[MARKET_FEATURES]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_scaled,
        y,
    )

    return scaler, model


def prepare_split(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:

    result = df.copy()

    if start is not None:
        result = result[
            result["timestamp"] >= start
        ]

    if end is not None:
        result = result[
            result["timestamp"] < end
        ]

    return result[
        result["target_class"].notna()
    ].copy()


def probabilities(
    scaler: StandardScaler,
    model: LogisticRegression,
    df: pd.DataFrame,
) -> np.ndarray:

    usable = df[
        MARKET_FEATURES
    ].dropna()

    if usable.empty:
        return np.empty(
            (0, len(model.classes_))
        )

    return model.predict_proba(
        scaler.transform(usable)
    )


def evaluate_selective(
    oos: pd.DataFrame,
    probabilities_array: np.ndarray,
    model_classes: np.ndarray,
    up_threshold: float,
    down_threshold: float,
) -> dict:

    usable = oos[
        MARKET_FEATURES
    ].dropna()

    if len(usable) != len(probabilities_array):
        raise RuntimeError(
            "Probability rows do not align "
            "with OOS observations."
        )

    class_to_index = {
        label: index
        for index, label
        in enumerate(model_classes)
    }

    up_probabilities = probabilities_array[
        :,
        class_to_index["UP"],
    ]

    down_probabilities = probabilities_array[
        :,
        class_to_index["DOWN"],
    ]

    predictions = np.full(
        len(probabilities_array),
        "NEUTRAL",
        dtype=object,
    )

    up_mask = (
        up_probabilities >= up_threshold
    )

    down_mask = (
        down_probabilities >= down_threshold
    )

    # Directional signal is emitted only when
    # one side clears its frozen threshold.
    # If both clear, choose the larger probability.
    both_mask = up_mask & down_mask

    predictions[
        up_mask & ~down_mask
    ] = "UP"

    predictions[
        down_mask & ~up_mask
    ] = "DOWN"

    predictions[
        both_mask
        & (
            up_probabilities
            >= down_probabilities
        )
    ] = "UP"

    predictions[
        both_mask
        & (
            down_probabilities
            > up_probabilities
        )
    ] = "DOWN"

    y_true = oos[
        "target_class"
    ].loc[
        usable.index
    ].to_numpy()

    directional_mask = (
        predictions != "NEUTRAL"
    )

    directional_count = int(
        directional_mask.sum()
    )

    total = len(predictions)

    coverage = (
        directional_count / total
        if total
        else 0.0
    )

    if directional_count:
        directional_accuracy = accuracy_score(
            y_true[directional_mask],
            predictions[directional_mask],
        )

        up_actual = (
            y_true == "UP"
        )

        up_predicted = (
            predictions == "UP"
        )

        down_actual = (
            y_true == "DOWN"
        )

        down_predicted = (
            predictions == "DOWN"
        )

        up_predictions = int(
            up_predicted.sum()
        )

        down_predictions = int(
            down_predicted.sum()
        )

        up_precision = (
            (
                (up_actual & up_predicted).sum()
                / up_predictions
            )
            if up_predictions
            else np.nan
        )

        down_precision = (
            (
                (down_actual & down_predicted).sum()
                / down_predictions
            )
            if down_predictions
            else np.nan
        )

    else:
        directional_accuracy = np.nan
        up_precision = np.nan
        down_precision = np.nan
    return {
        "coverage": coverage,
        "directional_count": directional_count,
        "neutral_count": int(
            (~directional_mask).sum()
        ),
        "directional_accuracy": (
            directional_accuracy
        ),
        "up_precision": up_precision,
        "down_precision": down_precision,
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
    }


def print_distribution(
    name: str,
    values: pd.Series,
) -> None:

    print()
    print(name)
    print("-" * 80)

    print(
        values.value_counts(
            normalize=True
        )
        .sort_index()
        .to_string()
    )


def main() -> None:

    print("=" * 80)
    print(
        "PHASE 2.2F — SELECTIVE PREDICTION"
    )
    print("=" * 80)

    market = load_market_data()

    data = prepare_dataset(
        market
    )

    training = prepare_split(
        data,
        None,
        TRAIN_END,
    )

    oos = prepare_split(
        data,
        TRAIN_END,
        FINAL_TEST_START,
    )

    locked = data[
        data["timestamp"] >= FINAL_TEST_START
    ]

    print()
    print(
        f"Input: {INPUT_PATH}"
    )
    print(
        f"Market rows: {len(market):,}"
    )
    print(
        f"Horizon: {HORIZON}"
    )
    print(
        f"Frozen target threshold: "
        f"±{THRESHOLD:.6%}"
    )
    print(
        f"Probability percentile: "
        f"P{PROBABILITY_PERCENTILE}"
    )

    print()
    print("=" * 80)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 80)

    print(
        f"Training: < {TRAIN_END}"
    )
    print(
        f"OOS: {TRAIN_END} → "
        f"{FINAL_TEST_START}"
    )
    print(
        f"Locked: >= {FINAL_TEST_START}"
    )

    print(
        f"\nTraining rows: {len(training):,}"
    )
    print(
        f"OOS rows:      {len(oos):,}"
    )
    print(
        f"Locked rows:   {len(locked):,}"
    )

    for asset in sorted(
        data["asset"].dropna().unique()
    ):

        print()
        print("=" * 80)
        print(f"ASSET: {asset}")
        print("=" * 80)

        asset_training = training[
            training["asset"] == asset
        ].copy()

        asset_oos = oos[
            oos["asset"] == asset
        ].copy()

        print(
            f"\nTraining target rows: "
            f"{len(asset_training):,}"
        )

        print(
            f"OOS target rows: "
            f"{len(asset_oos):,}"
        )

        print_distribution(
            "TRAINING TARGET DISTRIBUTION",
            asset_training[
                "target_class"
            ],
        )

        print_distribution(
            "OOS TARGET DISTRIBUTION",
            asset_oos[
                "target_class"
            ],
        )

        scaler, model = fit_model(
            asset_training
        )

        train_usable = asset_training[
            MARKET_FEATURES
        ].dropna()

        oos_usable = asset_oos[
            MARKET_FEATURES
        ].dropna()

        train_probabilities = model.predict_proba(
            scaler.transform(
                train_usable[
                    MARKET_FEATURES
                ]
            )
        )

        oos_probabilities = model.predict_proba(
            scaler.transform(
                oos_usable[
                    MARKET_FEATURES
                ]
            )
        )

        class_to_index = {
            label: index
            for index, label
            in enumerate(model.classes_)
        }

        up_index = class_to_index["UP"]
        down_index = class_to_index["DOWN"]

        train_up = train_probabilities[
            :,
            up_index,
        ]

        train_down = train_probabilities[
            :,
            down_index,
        ]

        up_threshold = np.percentile(
            train_up,
            PROBABILITY_PERCENTILE,
        )

        down_threshold = np.percentile(
            train_down,
            PROBABILITY_PERCENTILE,
        )

        print()
        print("FROZEN TRAINING PROBABILITY THRESHOLDS")
        print("-" * 80)

        print(
            f"P(UP) P{PROBABILITY_PERCENTILE}: "
            f"{up_threshold:.6f}"
        )

        print(
            f"P(DOWN) P{PROBABILITY_PERCENTILE}: "
            f"{down_threshold:.6f}"
        )

        result = evaluate_selective(
            asset_oos,
            oos_probabilities,
            model.classes_,
            up_threshold,
            down_threshold,
        )

        print()
        print("OOS SELECTIVE PERFORMANCE")
        print("-" * 80)

        print(
            f"Coverage:              "
            f"{result['coverage']:.4f}"
        )

        print(
            f"Directional signals:   "
            f"{result['directional_count']}"
        )

        print(
            f"Neutral / abstain:     "
            f"{result['neutral_count']}"
        )

        print(
            f"Directional accuracy:  "
            f"{result['directional_accuracy']:.4f}"
        )

        print(
            f"UP precision:          "
            f"{result['up_precision']:.4f}"
        )

        print(
            f"DOWN precision:        "
            f"{result['down_precision']:.4f}"
        )

        print(
            f"Macro-F1:              "
            f"{result['macro_f1']:.4f}"
        )

        print(
            f"Balanced accuracy:     "
            f"{result['balanced_accuracy']:.4f}"
        )

    print()
    print("=" * 80)
    print("LOCKED HOLDOUT PROTECTION")
    print("=" * 80)

    print(
        f"Locked rows available: "
        f"{len(locked):,}"
    )

    print(
        "Locked observations were not used "
        "for probability-threshold derivation "
        "or OOS evaluation."
    )

    print()
    print("=" * 80)
    print(
        "PHASE 2.2F DIAGNOSTIC COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()