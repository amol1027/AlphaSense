from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FEATURE_PATH = (
    "data/processed/phase1_features.csv"
)

FINAL_TEST_START = pd.Timestamp(
    "2026-08-10 00:00:00",
    tz="UTC",
)


MARKET_FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
]

PRICE_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

NEWS_FEATURES = [
    "sentiment_mean",
    "sentiment_std",
    "news_count",
    "positive_ratio",
    "negative_ratio",
]

REDDIT_FEATURES = [
    "reddit_sentiment_mean",
    "reddit_sentiment_std",
    "reddit_count",
    "reddit_positive_ratio",
    "reddit_negative_ratio",
    "reddit_score_mean",
    "reddit_comments_mean",
    "reddit_engagement_mean",
]


def safe_direction_accuracy(
    feature: pd.Series,
    target: pd.Series,
) -> float:
    """
    Measure whether the sign of a feature agrees with
    the binary target direction.

    This is only a simple univariate diagnostic.
    It is NOT a trading rule or model-selection step.
    """

    valid = (
        feature.notna()
        & target.notna()
    )

    if valid.sum() == 0:
        return np.nan

    x = pd.to_numeric(
        feature[valid],
        errors="coerce",
    )

    y = pd.to_numeric(
        target[valid],
        errors="coerce",
    )

    valid_numeric = (
        x.notna()
        & y.notna()
        & np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid_numeric]
    y = y[valid_numeric]

    if len(x) == 0:
        return np.nan

    prediction = (
        x > 0
    ).astype(int)

    return float(
        (prediction == y).mean()
    )


def feature_diagnostic(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    rows = []

    for feature in features:

        if feature not in df.columns:
            continue

        x = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        y_direction = pd.to_numeric(
            df["target_direction"],
            errors="coerce",
        )

        y_return = pd.to_numeric(
            df["target_return"],
            errors="coerce",
        )

        valid = (
            x.notna()
            & y_direction.notna()
            & y_return.notna()
            & np.isfinite(x)
            & np.isfinite(y_direction)
            & np.isfinite(y_return)
        )

        x_valid = x[valid]
        direction_valid = y_direction[valid]
        return_valid = y_return[valid]

        if len(x_valid) == 0:
            continue

        up = x_valid[
            direction_valid == 1
        ]

        down = x_valid[
            direction_valid == 0
        ]

        correlation = (
            x_valid.corr(
                return_valid
            )
        )

        direction_correlation = (
            x_valid.corr(
                direction_valid
            )
        )

        direction_accuracy = (
            safe_direction_accuracy(
                x_valid,
                direction_valid,
            )
        )

        rows.append(
            {
                "feature": feature,
                "samples": len(x_valid),
                "mean": x_valid.mean(),
                "std": x_valid.std(),
                "min": x_valid.min(),
                "max": x_valid.max(),
                "mean_up": (
                    up.mean()
                    if len(up)
                    else np.nan
                ),
                "mean_down": (
                    down.mean()
                    if len(down)
                    else np.nan
                ),
                "mean_difference": (
                    up.mean() - down.mean()
                    if len(up) and len(down)
                    else np.nan
                ),
                "corr_target_return": correlation,
                "corr_target_direction": (
                    direction_correlation
                ),
                "sign_accuracy": (
                    direction_accuracy
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "corr_target_direction",
            key=lambda x: x.abs(),
            ascending=False,
        ).reset_index(drop=True)

    return result


def print_section(
    title: str,
) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_feature_table(
    title: str,
    result: pd.DataFrame,
) -> None:

    print()
    print(title)
    print("-" * 70)

    if result.empty:
        print("No usable features.")
        return

    display_columns = [
        "feature",
        "samples",
        "mean_up",
        "mean_down",
        "mean_difference",
        "corr_target_return",
        "corr_target_direction",
        "sign_accuracy",
    ]

    print(
        result[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


def main() -> None:

    print("=" * 70)
    print("PHASE 1 FEATURE VS TARGET DIAGNOSTIC")
    print("=" * 70)

    print()
    print(
        "Input:",
        FEATURE_PATH,
    )

    print(
        "Final test starts:",
        FINAL_TEST_START,
    )

    df = pd.read_csv(
        FEATURE_PATH
    )

    df[
        "prediction_timestamp"
    ] = pd.to_datetime(
        df["prediction_timestamp"],
        utc=True,
        errors="coerce",
    )

    print()
    print(
        "Feature rows:",
        len(df),
    )

    print(
        "Assets:",
        df["asset"]
        .value_counts()
        .to_dict(),
    )

    # --------------------------------------------------
    # Training data only
    # --------------------------------------------------

    train = df[
        df["prediction_timestamp"]
        < FINAL_TEST_START
    ].copy()

    train = train[
        train["target_direction"].notna()
        & train["target_return"].notna()
    ].copy()

    print_section(
        "TRAINING DATA ONLY"
    )

    print(
        "Rows with valid targets:",
        len(train),
    )

    print(
        "Date range:",
        train["prediction_timestamp"].min(),
        "->",
        train["prediction_timestamp"].max(),
    )

    print()
    print(
        "Target distribution:"
    )

    print(
        train[
            "target_direction"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Positive rate:",
        f"{train['target_direction'].mean():.4f}",
    )

    # --------------------------------------------------
    # Asset-level diagnostics
    # --------------------------------------------------

    all_feature_groups = {
        "PRICE FEATURES": PRICE_FEATURES,
        "MARKET FEATURES": MARKET_FEATURES,
        "NEWS FEATURES": NEWS_FEATURES,
        "REDDIT FEATURES": REDDIT_FEATURES,
    }

    for asset in sorted(
        train["asset"].dropna().unique()
    ):

        asset_train = train[
            train["asset"] == asset
        ].copy()

        print_section(
            f"ASSET: {asset}"
        )

        print(
            "Training rows:",
            len(asset_train),
        )

        print(
            "Positive rate:",
            f"{asset_train['target_direction'].mean():.4f}",
        )

        for group_name, features in (
            all_feature_groups.items()
        ):

            result = feature_diagnostic(
                asset_train,
                features,
            )

            print_feature_table(
                group_name,
                result,
            )

    # --------------------------------------------------
    # Combined asset diagnostic
    # --------------------------------------------------

    print_section(
        "COMBINED ASSET DIAGNOSTIC"
    )

    combined_results = []

    for group_name, features in (
        all_feature_groups.items()
    ):

        result = feature_diagnostic(
            train,
            features,
        )

        if not result.empty:
            result = result.copy()
            result["group"] = group_name
            combined_results.append(
                result
            )

    if combined_results:

        combined = pd.concat(
            combined_results,
            ignore_index=True,
        )

        combined = combined.sort_values(
            "corr_target_direction",
            key=lambda x: x.abs(),
            ascending=False,
        )

        print(
            combined[
                [
                    "group",
                    "feature",
                    "samples",
                    "mean_difference",
                    "corr_target_return",
                    "corr_target_direction",
                    "sign_accuracy",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # --------------------------------------------------
    # Strongest univariate relationships
    # --------------------------------------------------

    print_section(
        "STRONGEST UNIVARIATE RELATIONSHIPS"
    )

    if combined_results:

        strongest = combined.sort_values(
            "corr_target_direction",
            key=lambda x: x.abs(),
            ascending=False,
        ).head(10)

        print(
            strongest[
                [
                    "group",
                    "feature",
                    "corr_target_direction",
                    "corr_target_return",
                    "sign_accuracy",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # --------------------------------------------------
    # Important warning
    # --------------------------------------------------

    print_section(
        "INTERPRETATION"
    )

    print(
        "This diagnostic uses TRAINING DATA ONLY."
    )

    print(
        "It does not tune thresholds, select models, "
        "or modify the frozen Phase 1 candidates."
    )

    print(
        "Correlation and sign accuracy are descriptive "
        "univariate diagnostics, not evidence of a "
        "profitable trading strategy."
    )

    print()
    print(
        "The locked final test remains untouched."
    )


if __name__ == "__main__":
    main()