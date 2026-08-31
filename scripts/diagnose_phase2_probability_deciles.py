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


def decile_table(
    probabilities: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:

    diagnostic = pd.DataFrame(
        {
            "probability": probabilities,
            "target": target,
        }
    ).dropna()

    diagnostic = diagnostic.sort_values(
        "probability"
    ).reset_index(drop=True)

    # qcut can fail if probabilities have many
    # identical values, so use rank first.
    diagnostic["rank"] = (
        diagnostic["probability"]
        .rank(method="first")
    )

    diagnostic["decile"] = pd.qcut(
        diagnostic["rank"],
        10,
        labels=False,
    ) + 1

    rows = []

    for decile, group in diagnostic.groupby(
        "decile"
    ):

        rows.append(
            {
                "decile": int(decile),
                "n": len(group),
                "mean_probability": (
                    group["probability"].mean()
                ),
                "actual_down_pct": (
                    (group["target"] == "DOWN").mean()
                ),
                "actual_neutral_pct": (
                    (group["target"] == "NEUTRAL").mean()
                ),
                "actual_up_pct": (
                    (group["target"] == "UP").mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def monotonicity_report(
    table: pd.DataFrame,
    probability_column: str,
) -> None:

    up_rates = table[
        "actual_up_pct"
    ].to_numpy()

    down_rates = table[
        "actual_down_pct"
    ].to_numpy()

    up_corr = np.corrcoef(
        table[probability_column],
        up_rates,
    )[0, 1]

    down_corr = np.corrcoef(
        table[probability_column],
        down_rates,
    )[0, 1]

    print(
        f"Probability vs UP-rate correlation: "
        f"{up_corr:.6f}"
    )

    print(
        f"Probability vs DOWN-rate correlation: "
        f"{down_corr:.6f}"
    )


def main() -> None:

    print("=" * 80)
    print(
        "PHASE 2.2E — PROBABILITY DECILE DIAGNOSTIC"
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
    print(
        f"Training rows: {len(train):,}"
    )
    print(
        f"OOS rows: {len(oos):,}"
    )
    print(
        f"Frozen threshold: ±{THRESHOLD:.6%}"
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

        probabilities = model.predict_proba(
            X_oos
        )

        classes = list(
            model.classes_
        )

        print()
        print("=" * 80)
        print(f"ASSET: {asset}")
        print("=" * 80)

        for label in [
            "DOWN",
            "UP",
        ]:

            if label not in classes:
                continue

            probability_index = classes.index(
                label
            )

            probability = pd.Series(
                probabilities[
                    :, probability_index
                ],
                index=oos_usable.index,
            )

            binary_target = (
                y_oos == label
            ).astype(int)

            auc = roc_auc_score(
                binary_target,
                probability,
            )

            print()
            print(
                f"{label} PROBABILITY DECILES"
            )
            print("-" * 80)

            print(
                f"One-vs-rest AUC: {auc:.6f}"
            )

            table = decile_table(
                probability,
                y_oos,
            )

            print(
                table.to_string(
                    index=False,
                    float_format=lambda x: (
                        f"{x:.6f}"
                    ),
                )
            )

            print()
            print(
                "MONOTONICITY"
            )
            print("-" * 80)

            monotonicity_report(
                table,
                "mean_probability",
            )

            # Explicit top-vs-bottom comparison.
            bottom = table.iloc[0]
            top = table.iloc[-1]

            print()
            print(
                "BOTTOM vs TOP DECILE"
            )
            print("-" * 80)

            print(
                f"Bottom mean P({label}): "
                f"{bottom['mean_probability']:.6f}"
            )

            print(
                f"Top mean P({label}): "
                f"{top['mean_probability']:.6f}"
            )

            print(
                f"Bottom actual UP rate: "
                f"{bottom['actual_up_pct']:.6f}"
            )

            print(
                f"Top actual UP rate: "
                f"{top['actual_up_pct']:.6f}"
            )

            print(
                f"Bottom actual DOWN rate: "
                f"{bottom['actual_down_pct']:.6f}"
            )

            print(
                f"Top actual DOWN rate: "
                f"{top['actual_down_pct']:.6f}"
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
        "Locked observations were not used "
        "for training or diagnosis."
    )

    print()
    print("=" * 80)
    print(
        "PHASE 2.2E DIAGNOSTIC COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()