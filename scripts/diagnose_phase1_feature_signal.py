from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


INPUT_PATH = Path(
    "data/processed/phase1_features.csv"
)

FINAL_TEST_START = pd.Timestamp(
    "2026-08-10",
    tz="UTC",
)

FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
    "sentiment_mean",
    "sentiment_std",
    "news_count",
    "positive_ratio",
    "negative_ratio",
]

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


def safe_auc(
    y: pd.Series,
    values: pd.Series,
) -> float:
    """
    Calculate ROC AUC safely.

    AUC is undefined when only one target class
    is present.
    """

    valid = (
        y.notna()
        & values.notna()
    )

    y_valid = y[valid]
    values_valid = values[valid]

    if y_valid.nunique() < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_valid,
            values_valid,
        )
    )


def correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    """
    Pearson correlation between a feature
    and binary target.
    """

    valid = (
        x.notna()
        & y.notna()
    )

    if valid.sum() < 2:
        return float("nan")

    return float(
        x[valid].corr(
            y[valid]
        )
    )


def describe_feature(
    df: pd.DataFrame,
    feature: str,
) -> dict:
    x = df[feature]
    y = df["target_direction"]

    valid = (
        x.notna()
        & y.notna()
    )

    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) == 0:
        return {
            "feature": feature,
            "samples": 0,
            "mean": np.nan,
            "std": np.nan,
            "corr": np.nan,
            "auc": np.nan,
            "mean_target_0": np.nan,
            "mean_target_1": np.nan,
            "mean_difference": np.nan,
        }

    mean_0 = (
        x_valid[y_valid == 0].mean()
    )

    mean_1 = (
        x_valid[y_valid == 1].mean()
    )

    return {
        "feature": feature,
        "samples": len(x_valid),
        "mean": x_valid.mean(),
        "std": x_valid.std(),
        "corr": correlation(
            x_valid,
            y_valid,
        ),
        "auc": safe_auc(
            y_valid,
            x_valid,
        ),
        "mean_target_0": mean_0,
        "mean_target_1": mean_1,
        "mean_difference": (
            mean_1 - mean_0
        ),
    }


def logistic_signal(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    usable = df[
        features
        + ["target_direction"]
    ].dropna()

    if usable.empty:
        return pd.DataFrame()

    if usable["target_direction"].nunique() < 2:
        return pd.DataFrame()

    X = usable[features]
    y = usable["target_direction"]

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

    coefficients = (
        model.coef_[0]
    )

    return pd.DataFrame(
        {
            "feature": features,
            "standardized_coefficient": coefficients,
            "absolute_coefficient": np.abs(
                coefficients
            ),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    )


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:

    print("=" * 70)
    print("PHASE 1 FEATURE-VS-TARGET SIGNAL DIAGNOSTIC")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    df[
        "prediction_timestamp"
    ] = pd.to_datetime(
        df[
            "prediction_timestamp"
        ],
        utc=True,
    )

    print(
        f"Input: {INPUT_PATH}"
    )

    print(
        f"Feature rows: {len(df):,}"
    )

    print(
        f"Final test starts: "
        f"{FINAL_TEST_START}"
    )

    print_section(
        "TRAINING PERIOD"
    )

    train = df[
        df["prediction_timestamp"]
        < FINAL_TEST_START
    ].copy()

    print(
        f"Training rows: {len(train):,}"
    )

    print()

    for asset in sorted(
        train["asset"].dropna().unique()
    ):

        asset_df = train[
            train["asset"] == asset
        ].copy()

        print_section(
            f"ASSET: {asset}"
        )

        print(
            f"Training rows: "
            f"{len(asset_df):,}"
        )

        target = (
            asset_df[
                "target_direction"
            ]
        )

        print(
            "Target distribution:"
        )

        print(
            target.value_counts(
                dropna=False
            ).sort_index().to_string()
        )

        valid_target = target.dropna()

        print()

        print(
            f"Usable target rows: "
            f"{len(valid_target):,}"
        )

        if not valid_target.empty:
            print(
                f"Positive rate: "
                f"{valid_target.mean():.4f}"
            )

        print()

        print(
            "FEATURE SIGNAL"
        )

        print(
            "-" * 70
        )

        rows = []

        for feature in FEATURES:

            if feature not in asset_df.columns:
                continue

            rows.append(
                describe_feature(
                    asset_df,
                    feature,
                )
            )

        result = pd.DataFrame(
            rows
        )

        if not result.empty:

            display_columns = [
                "feature",
                "samples",
                "corr",
                "auc",
                "mean_target_0",
                "mean_target_1",
                "mean_difference",
            ]

            print(
                result[
                    display_columns
                ].to_string(
                    index=False,
                    float_format=lambda x: (
                        f"{x:.6f}"
                    ),
                )
            )

        print_section(
            f"{asset} — MARKET FEATURE LOGISTIC SIGNAL"
        )

        market_result = logistic_signal(
            asset_df,
            MARKET_FEATURES,
        )

        if market_result.empty:
            print(
                "Insufficient data."
            )
        else:
            print(
                market_result.to_string(
                    index=False,
                    float_format=lambda x: (
                        f"{x:.6f}"
                    ),
                )
            )

        print_section(
            f"{asset} — MARKET + NEWS LOGISTIC SIGNAL"
        )

        combined_result = logistic_signal(
            asset_df,
            MARKET_FEATURES
            + NEWS_FEATURES,
        )

        if combined_result.empty:
            print(
                "Insufficient data."
            )
        else:
            print(
                combined_result.to_string(
                    index=False,
                    float_format=lambda x: (
                        f"{x:.6f}"
                    ),
                )
            )

    print_section(
        "TEST PERIOD EXCLUSION CHECK"
    )

    test_rows = df[
        df["prediction_timestamp"]
        >= FINAL_TEST_START
    ]

    print(
        f"Locked test rows excluded "
        f"from feature-signal fitting: "
        f"{len(test_rows):,}"
    )

    print(
        "Diagnostic uses training period only."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This diagnostic is descriptive."
    )

    print(
        "It must not be used to select or tune "
        "the frozen Phase 1 candidates."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()