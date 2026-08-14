import pandas as pd

from src.ingestion.loader import load_news
from src.sentiment.dummy import DummySentimentProvider
from src.features.sentiment_features import aggregate_sentiment


def main():
    # 1. Load and validate news
    articles = load_news("data/raw/news_sample.csv")

    print(f"Loaded {len(articles)} articles.")

    # 2. Initialize sentiment provider
    sentiment_provider = DummySentimentProvider()

    # 3. Generate article-level sentiment
    sentiment_records = []

    for article in articles:
        result = sentiment_provider.predict(article.text)

        sentiment_records.append(
            {
                "asset": article.asset,
                "published_at": article.published_at,
                "positive_probability": result.positive_probability,
                "neutral_probability": result.neutral_probability,
                "negative_probability": result.negative_probability,
                "sentiment_score": result.sentiment_score,
            }
        )

    sentiment_df = pd.DataFrame(sentiment_records)

    print(f"Generated {len(sentiment_df)} sentiment records.")

    # 4. Aggregate sentiment
    feature_df = aggregate_sentiment(sentiment_df)

    print("\nFinal feature table:")
    print(feature_df)

    # 5. Save processed features
    output_path = "data/processed/sentiment_features.csv"
    feature_df.to_csv(output_path, index=False)

    print(f"\nSaved features to: {output_path}")


if __name__ == "__main__":
    main()