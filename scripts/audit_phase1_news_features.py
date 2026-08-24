from pathlib import Path

import pandas as pd

from src.features.sentiment_features import (
    aggregate_sentiment,
)


FEATURE_PATH = Path(
    "data/processed/phase1_features.csv"
)

SENTIMENT_PATH = Path(
    "data/processed/research_news_sentiment.csv"
)

NEWS_PATH = Path(
    "data/raw/news/research_news.csv"
)

FINAL_TEST_START = pd.Timestamp(
    "2026-08-10",
    tz="UTC",
)

NEWS_LOOKBACK = pd.Timedelta(
    hours=1
)


REQUIRED_SENTIMENT_COLUMNS = {
    "asset",
    "published_at",
    "sentiment_score",
    "positive_probability",
    "negative_probability",
}


NEWS_FEATURE_COLUMNS = [
    "sentiment_mean",
    "sentiment_std",
    "news_count",
    "positive_ratio",
    "negative_ratio",
]


def normalize_timestamp(
    values,
) -> pd.Series:
    return pd.to_datetime(
        values,
        utc=True,
        errors="coerce",
    )


def calculate_expected_news_features(
    predictions: pd.DataFrame,
    sentiments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Independently reconstruct the news sentiment
    features using the same information cutoff:

        prediction - 1 hour <= published_at
        <= prediction

    Only matching assets are eligible.

    The calculation is performed from the complete
    sentiment dataset before comparison.
    """

    rows = []

    for _, prediction in predictions.iterrows():

        prediction_timestamp = (
            prediction["prediction_timestamp"]
        )

        window_start = (
            prediction_timestamp
            - NEWS_LOOKBACK
        )

        eligible = sentiments[
            (
                sentiments["asset"]
                == prediction["asset"]
            )
            & (
                sentiments["published_at"]
                >= window_start
            )
            & (
                sentiments["published_at"]
                <= prediction_timestamp
            )
        ].copy()

        if eligible.empty:
            continue

        for _, article in eligible.iterrows():

            rows.append(
                {
                    "asset": prediction["asset"],
                    "exchange": prediction["exchange"],
                    "prediction_timestamp": (
                        prediction_timestamp
                    ),
                    "sentiment_score": (
                        article["sentiment_score"]
                    ),
                    "positive_probability": (
                        article[
                            "positive_probability"
                        ]
                    ),
                    "negative_probability": (
                        article[
                            "negative_probability"
                        ]
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "asset",
                "exchange",
                "prediction_timestamp",
                *NEWS_FEATURE_COLUMNS,
            ]
        )

    sentiment_rows = pd.DataFrame(rows)

    return aggregate_sentiment(
        sentiment_rows
    )


def main() -> None:

    print("=" * 70)
    print("PHASE 1 NEWS FEATURE CUTOFF AUDIT")
    print("=" * 70)

    for path in [
        FEATURE_PATH,
        SENTIMENT_PATH,
        NEWS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    features = pd.read_csv(
        FEATURE_PATH
    )

    sentiments = pd.read_csv(
        SENTIMENT_PATH
    )

    news = pd.read_csv(
        NEWS_PATH
    )

    features[
        "prediction_timestamp"
    ] = normalize_timestamp(
        features["prediction_timestamp"]
    )

    sentiments[
        "published_at"
    ] = normalize_timestamp(
        sentiments["published_at"]
    )

    news[
        "published_at"
    ] = normalize_timestamp(
        news["published_at"]
    )

    missing = (
        REQUIRED_SENTIMENT_COLUMNS
        - set(sentiments.columns)
    )

    if missing:
        raise ValueError(
            "Missing sentiment columns: "
            f"{sorted(missing)}"
        )

    print()
    print(
        f"Feature rows:             "
        f"{len(features):,}"
    )

    print(
        f"Sentiment rows:            "
        f"{len(sentiments):,}"
    )

    print(
        f"Raw news rows:             "
        f"{len(news):,}"
    )

    failures = []

    # ==============================================================
    # 1. Check sentiment timestamps
    # ==============================================================

    print()
    print("=" * 70)
    print("SENTIMENT DATA QUALITY")
    print("=" * 70)

    invalid_sentiment_timestamps = int(
        sentiments[
            "published_at"
        ].isna().sum()
    )

    print(
        "Invalid sentiment timestamps:",
        invalid_sentiment_timestamps,
    )

    if invalid_sentiment_timestamps:
        failures.append(
            "Sentiment dataset contains invalid "
            "published_at timestamps."
        )

    # ==============================================================
    # 2. Restrict predictions to final test period
    # ==============================================================

    test_predictions = features[
        features["prediction_timestamp"]
        >= FINAL_TEST_START
    ][
        [
            "asset",
            "exchange",
            "prediction_timestamp",
        ]
    ].drop_duplicates()

    print()
    print(
        "Final-test prediction timestamps:",
        len(test_predictions),
    )

    # ==============================================================
    # 3. Independently reconstruct news features
    # ==============================================================

    expected = calculate_expected_news_features(
        test_predictions,
        sentiments,
    )

    # ==============================================================
    # 4. Compare stored features against independent
    #    reconstruction
    # ==============================================================

    actual = features[
        features["prediction_timestamp"]
        >= FINAL_TEST_START
    ][
        [
            "asset",
            "exchange",
            "prediction_timestamp",
            *NEWS_FEATURE_COLUMNS,
        ]
    ].copy()

    merged = actual.merge(
        expected,
        on=[
            "asset",
            "exchange",
            "prediction_timestamp",
        ],
        how="left",
        suffixes=(
            "_actual",
            "_expected",
        ),
    )

    # Predictions with no eligible news legitimately
    # have zero-valued news features in the feature dataset.

    for column in NEWS_FEATURE_COLUMNS:

        actual_column = (
            f"{column}_actual"
        )

        expected_column = (
            f"{column}_expected"
        )

        expected_values = (
            merged[expected_column]
            .fillna(0.0)
        )

        actual_values = pd.to_numeric(
            merged[actual_column],
            errors="coerce",
        )

        differences = (
            actual_values
            - expected_values
        ).abs()

        valid = (
            actual_values.notna()
            & expected_values.notna()
        )

        if not valid.any():
            print(
                f"{column:24s}"
                " no comparable values"
            )
            continue

        max_difference = (
            differences[valid].max()
        )

        print(
            f"{column:24s}"
            f" max difference: "
            f"{max_difference:.12f}"
        )

        if max_difference > 1e-10:
            failures.append(
                f"{column} differs from "
                "independent news reconstruction."
            )

    # ==============================================================
    # 5. Explicit future-news leakage test
    # ==============================================================

    print()
    print("=" * 70)
    print("FUTURE NEWS CUTOFF TEST")
    print("=" * 70)

    future_news_violations = 0

    for _, prediction in test_predictions.iterrows():

        prediction_timestamp = (
            prediction[
                "prediction_timestamp"
            ]
        )

        future_articles = sentiments[
            (
                sentiments["asset"]
                == prediction["asset"]
            )
            & (
                sentiments["published_at"]
                > prediction_timestamp
            )
        ]

        if future_articles.empty:
            continue

        # Check whether the stored feature is non-zero
        # when all future information is excluded.
        actual_row = actual[
            (
                actual["asset"]
                == prediction["asset"]
            )
            & (
                actual[
                    "prediction_timestamp"
                ]
                == prediction_timestamp
            )
        ]

        if actual_row.empty:
            continue

        # Reconstruct features using only information
        # available at prediction time.
        available = sentiments[
            (
                sentiments["asset"]
                == prediction["asset"]
            )
            & (
                sentiments["published_at"]
                <= prediction_timestamp
            )
            & (
                sentiments["published_at"]
                >= (
                    prediction_timestamp
                    - NEWS_LOOKBACK
                )
            )
        ]

        if available.empty:
            expected_count = 0
        else:
            expected_count = len(
                available
            )

        stored_count = int(
            actual_row.iloc[0][
                "news_count"
            ]
        )

        if stored_count != expected_count:
            future_news_violations += 1

    print(
        "Future-news cutoff violations:",
        future_news_violations,
    )

    if future_news_violations:
        failures.append(
            "Future news appears to affect "
            "stored prediction features."
        )

    # ==============================================================
    # 6. Boundary spot checks
    # ==============================================================

    print()
    print("=" * 70)
    print("BOUNDARY SPOT CHECKS")
    print("=" * 70)

    for asset in sorted(
        test_predictions["asset"].unique()
    ):

        print()
        print(
            f"ASSET: {asset}"
        )

        sample = (
            actual[
                actual["asset"] == asset
            ]
            .sort_values(
                "prediction_timestamp"
            )
            [
                [
                    "prediction_timestamp",
                    "sentiment_mean",
                    "sentiment_std",
                    "news_count",
                    "positive_ratio",
                    "negative_ratio",
                ]
            ]
            .head(10)
        )

        print(
            sample.to_string(
                index=False
            )
        )

    # ==============================================================
    # Final status
    # ==============================================================

    print()
    print("=" * 70)

    if failures:
        print("AUDIT: FAIL")
        print("=" * 70)

        for failure in failures:
            print(
                f"- {failure}"
            )

        raise SystemExit(1)

    print("AUDIT: PASS")
    print("=" * 70)

    print()
    print(
        "News sentiment features match the "
        "independent one-hour information-cutoff "
        "reconstruction."
    )

    print(
        "No future-news leakage was detected."
    )


if __name__ == "__main__":
    main()