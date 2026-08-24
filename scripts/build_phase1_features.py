from pathlib import Path

import pandas as pd

from src.features.build_features import build_hourly_features


MARKET_PATH = (
    "data/raw/market/phase1_research_market_15m.csv"
)

NEWS_PATH = (
    "data/raw/news/research_news.csv"
)

REDDIT_PATH = (
    "data/raw/reddit_sample.csv"
)

NEWS_SENTIMENT_PATH = (
    "data/processed/research_news_sentiment.csv"
)

OUTPUT_PATH = (
    "data/processed/phase1_features.csv"
)


def main() -> None:
    print("=" * 70)
    print("PHASE 1 FEATURE DATASET BUILD")
    print("=" * 70)

    print()
    print("Market:", MARKET_PATH)
    print("News:", NEWS_PATH)
    print("Reddit:", REDDIT_PATH)
    print("Sentiment:", NEWS_SENTIMENT_PATH)
    print()

    features = build_hourly_features(
        MARKET_PATH,
        NEWS_PATH,
        REDDIT_PATH,
        news_sentiment_path=(
            NEWS_SENTIMENT_PATH
        ),
    )

    features["prediction_timestamp"] = (
        pd.to_datetime(
            features["prediction_timestamp"],
            utc=True,
        )
    )

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Feature rows:", f"{len(features):,}")

    print()
    print("Assets:")
    print(
        features["asset"]
        .value_counts()
        .to_string()
    )

    print()
    print("Columns:")
    print(features.columns.tolist())

    print()
    print("Prediction range:")
    print(
        features["prediction_timestamp"].min(),
        "->",
        features["prediction_timestamp"].max(),
    )

    print()
    print("Target coverage:")
    print(
        "Valid target:",
        features["target_direction"]
        .notna()
        .sum(),
    )
    print(
        "Missing target:",
        features["target_direction"]
        .isna()
        .sum(),
    )

    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()