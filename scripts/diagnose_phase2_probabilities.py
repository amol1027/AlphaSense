from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

OOS_START = pd.Timestamp(
    "2026-07-01",
    tz="UTC",
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

    return add_three_class_target(
        features
    )


def safe_auc(
    y_binary: pd.Series,
    probabilities: np.ndarray,
) -> float:

    if y_binary.nunique() < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_binary,
            probabilities,
        )
    )


def describe_probability(
    probabilities: np.ndarray,
    label: str,
) -> None:

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    print()
    print(label)
    print("-" * 70)

    print(
        f"Min:    {probabilities.min():.6f}"
    )
    print(
        f"P25:    {np.percentile(probabilities, 25):.6f}"
    )
    print(
        f"Median: {np.median(probabilities):.6f}"
    )
    print(
        f"P75:    {np.percentile(probabilities, 75):.6f}"
    )
    print(
        f"Max:    {probabilities.max():.6f}"
    )
    print(
        f"Mean:   {probabilities.mean():.6f}"
    )
    print(
        f"Std:    {probabilities.std():.6f}"
    )


def main() -> None:

    print("=" * 80)
    print(
        "PHASE 2.2D — THREE-CLASS PROBABILITY DIAGNOSTIC"
    )
    print("=" * 80)

    market = load_market_data()

    data = prepare_dataset(
        market
    )

    train = data[
        data["timestamp"] < OOS_START
    ].copy()

    oos = data[
        (data["timestamp"] >= OOS_START)
        & (
            data["timestamp"]
            < FINAL_TEST_START
        )
    ].copy()

    train = train[
        train["target_class"].notna()
    ].copy()

    oos = oos[
        oos["target_class"].notna()
    ].copy()

    print()
    print(f"Input: {INPUT_PATH}")
    print(f"Market rows: {len(market):,}")
    print(
        f"Training rows: {len(train):,}"
    )
    print(
        f"OOS rows: {len(oos):,}"
    )
    print(
        f"Locked test starts: "
        f"{FINAL_TEST_START}"
    )

    for asset in sorted(
        train["asset"].unique()
    ):

        train_asset = train[
            train["asset"] == asset
        ].copy()

        oos_asset = oos[
            oos["asset"] == asset
        ].copy()

        train_usable = train_asset[
            MARKET_FEATURES
            + ["target_class"]
        ].dropna()

        oos_usable = oos_asset[
            MARKET_FEATURES
            + ["target_class"]
        ].dropna()

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            train_usable[MARKET_FEATURES]
        )

        X_oos = scaler.transform(
            oos_usable[MARKET_FEATURES]
        )

        y_train = train_usable[
            "target_class"
        ]

        y_oos = oos_usable[
            "target_class"
        ]

        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

        model.fit(
            X_train,
            y_train,
        )

        train_probabilities = (
            model.predict_proba(
                X_train
            )
        )

        oos_probabilities = (
            model.predict_proba(
                X_oos
            )
        )

        classes = list(
            model.classes_
        )

        print()
        print("=" * 80)
        print(f"ASSET: {asset}")
        print("=" * 80)

        print()
        print("MODEL CLASSES")
        print("-" * 80)
        print(classes)

        # ----------------------------------------------------------
        # Probability summaries
        # ----------------------------------------------------------

        for label in LABELS:

            if label not in classes:
                continue

            index = classes.index(label)

            describe_probability(
                train_probabilities[:, index],
                f"TRAIN P({label})",
            )

            describe_probability(
                oos_probabilities[:, index],
                f"OOS P({label})",
            )

        # ----------------------------------------------------------
        # Probability by actual class
        # ----------------------------------------------------------

        print()
        print("OOS PROBABILITY BY ACTUAL CLASS")
        print("-" * 80)

        diagnostic = pd.DataFrame(
            {
                "actual": y_oos.to_numpy(),
            }
        )

        for label in LABELS:

            if label not in classes:
                continue

            index = classes.index(label)

            diagnostic[
                f"p_{label}"
            ] = oos_probabilities[:, index]

        for actual in LABELS:

            subset = diagnostic[
                diagnostic["actual"] == actual
            ]

            if subset.empty:
                continue

            print()
            print(
                f"Actual {actual} "
                f"(n={len(subset)})"
            )

            for label in LABELS:

                column = f"p_{label}"

                print(
                    f"  mean P({label}): "
                    f"{subset[column].mean():.6f}"
                )

                print(
                    f"  median P({label}): "
                    f"{subset[column].median():.6f}"
                )

        # ----------------------------------------------------------
        # One-vs-rest AUC
        # ----------------------------------------------------------

        print()
        print("OOS ONE-VS-REST AUC")
        print("-" * 80)

        for label in LABELS:

            if label not in classes:
                continue

            index = classes.index(label)

            y_binary = (
                y_oos == label
            ).astype(int)

            auc = safe_auc(
                y_binary,
                oos_probabilities[:, index],
            )

            print(
                f"{label:<8}: "
                f"{auc:.6f}"
            )

        # ----------------------------------------------------------
        # Prediction distribution
        # ----------------------------------------------------------

        predictions = model.predict(
            X_oos
        )

        print()
        print("OOS PREDICTION DISTRIBUTION")
        print("-" * 80)

        prediction_counts = (
            pd.Series(predictions)
            .value_counts()
            .reindex(LABELS)
            .fillna(0)
            .astype(int)
        )

        total = prediction_counts.sum()

        for label in LABELS:

            count = prediction_counts[
                label
            ]

            print(
                f"{label:<8}: "
                f"{count:>5} "
                f"({count / total:.2%})"
            )

    print()
    print("=" * 80)
    print("LOCKED HOLDOUT PROTECTION")
    print("=" * 80)

    locked_rows = data[
        data["timestamp"] >= FINAL_TEST_START
    ]

    print(
        f"Locked rows present: "
        f"{len(locked_rows):,}"
    )

    print(
        "The locked holdout was not used for "
        "training, probability analysis, or "
        "model selection."
    )

    print()
    print("=" * 80)
    print(
        "PHASE 2.2D DIAGNOSTIC COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()