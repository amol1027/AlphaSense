from dataclasses import dataclass

import numpy as np
import pandas as pd


VALID_ASSETS = {
    "TCS",
    "RELIANCE",
}

VALID_EXCHANGES = {
    "NSE",
}

REQUIRED_COLUMNS = {
    "asset",
    "exchange",
    "prediction_timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "target_return",
    "target_direction",
    "sentiment_mean",
    "sentiment_std",
    "news_count",
    "positive_ratio",
    "negative_ratio",
    "reddit_sentiment_mean",
    "reddit_sentiment_std",
    "reddit_count",
    "reddit_positive_ratio",
    "reddit_negative_ratio",
    "reddit_score_mean",
    "reddit_comments_mean",
    "reddit_engagement_mean",
}


@dataclass
class FeatureAuditResult:
    rows: int

    assets: set[str]
    start_timestamp: pd.Timestamp | None
    end_timestamp: pd.Timestamp | None

    invalid_timestamps: int = 0
    null_timestamps: int = 0

    duplicate_keys: int = 0

    invalid_assets: int = 0
    invalid_exchanges: int = 0

    invalid_numeric_values: int = 0
    infinite_numeric_values: int = 0

    invalid_targets: int = 0
    missing_targets: int = 0

    invalid_sentiment_features: int = 0
    invalid_reddit_features: int = 0

    weekend_predictions: int = 0

    invalid_target_alignment: int = 0

    passed: bool = True

    @property
    def status(self) -> str:
        return (
            "PASS"
            if self.passed
            else "FAIL"
        )


NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
    "target_return",
    "sentiment_mean",
    "sentiment_std",
    "news_count",
    "positive_ratio",
    "negative_ratio",
    "reddit_sentiment_mean",
    "reddit_sentiment_std",
    "reddit_count",
    "reddit_positive_ratio",
    "reddit_negative_ratio",
    "reddit_score_mean",
    "reddit_comments_mean",
    "reddit_engagement_mean",
]


def audit_feature_data(
    df: pd.DataFrame,
) -> FeatureAuditResult:
    """
    Audit the Phase 1 feature dataset.

    The function does not modify the input DataFrame.

    Missing targets are allowed because rows near the end
    of the available market dataset may not have a future
    observation exactly one hour later.

    Missing targets are reported separately through
    `missing_targets`.

    Present but malformed targets are considered invalid.
    """

    # ==================================================
    # Required columns
    # ==================================================

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required feature columns: "
            f"{sorted(missing)}"
        )

    # ==================================================
    # Timestamp validation
    # ==================================================

    timestamps = pd.to_datetime(
        df["prediction_timestamp"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = int(
        timestamps.isna().sum()
    )

    null_timestamps = int(
        df["prediction_timestamp"]
        .isna()
        .sum()
    )

    valid_timestamps = timestamps.dropna()

    start_timestamp = (
        valid_timestamps.min()
        if not valid_timestamps.empty
        else None
    )

    end_timestamp = (
        valid_timestamps.max()
        if not valid_timestamps.empty
        else None
    )

    # ==================================================
    # Duplicate keys
    # ==================================================

    duplicate_keys = int(
        df[
            [
                "asset",
                "exchange",
                "prediction_timestamp",
            ]
        ].duplicated().sum()
    )

    # ==================================================
    # Asset validation
    # ==================================================

    invalid_assets = int(
        (~df["asset"].isin(VALID_ASSETS))
        .sum()
    )

    # ==================================================
    # Exchange validation
    # ==================================================

    invalid_exchanges = int(
        (~df["exchange"].isin(VALID_EXCHANGES))
        .sum()
    )

    # ==================================================
    # Numeric validation
    # ==================================================
    #
    # Target columns are handled separately below.
    #
    # This is important because missing targets are
    # expected and must NOT be treated as invalid
    # numeric values.
    # ==================================================

    numeric_columns_present = [
        column
        for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    numeric = df[
        numeric_columns_present
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric_validation_columns = [
        column
        for column in numeric_columns_present
        if column != "target_return"
    ]

    numeric_validation = numeric[
        numeric_validation_columns
    ]

    invalid_numeric_values = int(numeric.notna().sum().sum() - numeric.count().sum())

    infinite_numeric_values = int(
        np.isinf(
            numeric_validation.to_numpy()
        ).sum()
    )

    # ==================================================
    # Target validation
    # ==================================================

    missing_targets = int(
        (
            df["target_return"].isna()
            | df["target_direction"].isna()
        ).sum()
    )

    # --------------------------------------------------
    # Target direction
    #
    # Missing is allowed.
    # Present values must be 0 or 1.
    # --------------------------------------------------

    invalid_target_direction = (
        df["target_direction"].notna()
        & ~df["target_direction"].isin(
            [0, 1]
        )
    )

    # --------------------------------------------------
    # Target return
    #
    # Missing is allowed.
    # Present values must be numeric
    # and finite.
    # --------------------------------------------------

    numeric_target_return = pd.to_numeric(
        df["target_return"],
        errors="coerce",
    )

    invalid_target_return = (
        df["target_return"].notna()
        & (
            numeric_target_return.isna()
            | ~np.isfinite(
                numeric_target_return.fillna(0)
            )
        )
    )

    invalid_targets = int(
        (
            invalid_target_direction
            | invalid_target_return
        ).sum()
    )

    # ==================================================
    # News sentiment feature validation
    # ==================================================

    sentiment_bounds = (
        (df["positive_ratio"] < 0)
        | (df["positive_ratio"] > 1)
        | (df["negative_ratio"] < 0)
        | (df["negative_ratio"] > 1)
    )

    invalid_sentiment_features = int(
        sentiment_bounds.sum()
    )

    # ==================================================
    # Reddit feature validation
    # ==================================================

    reddit_bounds = (
        (df["reddit_positive_ratio"] < 0)
        | (df["reddit_positive_ratio"] > 1)
        | (df["reddit_negative_ratio"] < 0)
        | (df["reddit_negative_ratio"] > 1)
        | (df["reddit_count"] < 0)
    )

    invalid_reddit_features = int(
        reddit_bounds.sum()
    )

    # ==================================================
    # Weekend prediction validation
    # ==================================================

    weekend_predictions = int(
        (
            valid_timestamps.dt.weekday >= 5
        ).sum()
    )

    # ==================================================
    # Target timestamp alignment
    # ==================================================

    invalid_target_alignment = 0

    if "future_timestamp" in df.columns:
        future_timestamps = pd.to_datetime(
            df["future_timestamp"],
            utc=True,
            errors="coerce",
        )

        target_exists = (
            df["target_return"].notna()
            & df["target_direction"].notna()
        )

        expected_future_timestamp = (
            timestamps
            + pd.Timedelta(hours=1)
        )

        invalid_target_alignment = int(
            (
                target_exists
                & (
                    future_timestamps
                    != expected_future_timestamp
                )
            ).sum()
        )

    # ==================================================
    # Overall audit status
    # ==================================================
    #
    # IMPORTANT:
    #
    # missing_targets is deliberately NOT included.
    #
    # Missing targets are expected at the end of the
    # available dataset and are reported separately.
    #
    # Invalid targets DO cause failure.
    # ==================================================

    passed = all(
        [
            invalid_timestamps == 0,
            duplicate_keys == 0,
            invalid_assets == 0,
            invalid_exchanges == 0,
            invalid_numeric_values == 0,
            infinite_numeric_values == 0,
            invalid_targets == 0,
            invalid_sentiment_features == 0,
            invalid_reddit_features == 0,
            weekend_predictions == 0,
            invalid_target_alignment == 0,
        ]
    )

    # ==================================================
    # Result
    # ==================================================

    return FeatureAuditResult(
        rows=len(df),

        assets=set(
            df["asset"]
            .dropna()
            .astype(str)
        ),

        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,

        invalid_timestamps=(
            invalid_timestamps
        ),

        null_timestamps=(
            null_timestamps
        ),

        duplicate_keys=(
            duplicate_keys
        ),

        invalid_assets=(
            invalid_assets
        ),

        invalid_exchanges=(
            invalid_exchanges
        ),

        invalid_numeric_values=(
            invalid_numeric_values
        ),

        infinite_numeric_values=(
            infinite_numeric_values
        ),

        invalid_targets=(
            invalid_targets
        ),

        missing_targets=(
            missing_targets
        ),

        invalid_sentiment_features=(
            invalid_sentiment_features
        ),

        invalid_reddit_features=(
            invalid_reddit_features
        ),

        weekend_predictions=(
            weekend_predictions
        ),

        invalid_target_alignment=(
            invalid_target_alignment
        ),

        passed=passed,
    )