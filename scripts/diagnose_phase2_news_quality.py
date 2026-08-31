from pathlib import Path

import numpy as np
import pandas as pd


MARKET_PATH = Path(
    "data/raw/market/phase1_research_market_15m.csv"
)

NEWS_PATH = Path(
    "data/processed/research_news_sentiment.csv"
)

TRAIN_END = pd.Timestamp(
    "2026-07-01 00:00:00",
    tz="UTC",
)

LOCKED_START = pd.Timestamp(
    "2026-08-10 00:00:00",
    tz="UTC",
)

HORIZON = pd.Timedelta(hours=1)

TARGET_THRESHOLD = 0.00203666

NEWS_WINDOW = pd.Timedelta(minutes=60)

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


def load_market() -> pd.DataFrame:

    if not MARKET_PATH.exists():
        raise FileNotFoundError(
            f"Market file not found: {MARKET_PATH}"
        )

    market = pd.read_csv(
        MARKET_PATH
    )

    required = {
        "asset",
        "exchange",
        "timestamp",
        "close",
    }

    missing = required - set(
        market.columns
    )

    if missing:
        raise ValueError(
            "Missing market columns: "
            f"{sorted(missing)}"
        )

    market["timestamp"] = pd.to_datetime(
        market["timestamp"],
        utc=True,
    )

    duplicate_count = market.duplicated(
        [
            "asset",
            "exchange",
            "timestamp",
        ]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} "
            "duplicate market observations."
        )

    return (
        market.sort_values(
            [
                "asset",
                "exchange",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )


def load_news() -> pd.DataFrame:

    if not NEWS_PATH.exists():
        raise FileNotFoundError(
            f"News file not found: {NEWS_PATH}"
        )

    news = pd.read_csv(
        NEWS_PATH
    )

    required = {
        "asset",
        "published_at",
        "sentiment_score",
        "positive_probability",
        "negative_probability",
    }

    missing = required - set(
        news.columns
    )

    if missing:
        raise ValueError(
            "Missing news columns: "
            f"{sorted(missing)}"
        )

    news["published_at"] = pd.to_datetime(
        news["published_at"],
        utc=True,
    )

    return (
        news.sort_values(
            [
                "asset",
                "published_at",
            ]
        )
        .reset_index(drop=True)
    )


def build_target(
    market: pd.DataFrame,
) -> pd.DataFrame:

    result = market.copy()

    result["future_close"] = (
        result
        .groupby(
            ["asset", "exchange"]
        )["close"]
        .shift(-4)
    )

    result["target_return"] = (
        result["future_close"]
        - result["close"]
    ) / result["close"]

    result["target_class"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )

    returns = result["target_return"]

    result.loc[
        returns < -TARGET_THRESHOLD,
        "target_class",
    ] = "DOWN"

    result.loc[
        returns.abs() <= TARGET_THRESHOLD,
        "target_class",
    ] = "NEUTRAL"

    result.loc[
        returns > TARGET_THRESHOLD,
        "target_class",
    ] = "UP"

    return result


def aggregate_news(
    predictions: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, prediction in predictions.iterrows():

        asset = prediction["asset"]
        timestamp = prediction["timestamp"]

        window_start = (
            timestamp - NEWS_WINDOW
        )

        eligible = news[
            (news["asset"] == asset)
            & (
                news["published_at"]
                >= window_start
            )
            & (
                news["published_at"]
                <= timestamp
            )
        ]

        if eligible.empty:
            rows.append(
                {
                    "asset": asset,
                    "exchange": prediction[
                        "exchange"
                    ],
                    "timestamp": timestamp,
                    "sentiment_mean": np.nan,
                    "sentiment_std": np.nan,
                    "news_count": np.nan,
                    "positive_ratio": np.nan,
                    "negative_ratio": np.nan,
                }
            )

            continue

        sentiment = (
            eligible["sentiment_score"]
        )

        rows.append(
            {
                "asset": asset,
                "exchange": prediction[
                    "exchange"
                ],
                "timestamp": timestamp,
                "sentiment_mean": (
                    sentiment.mean()
                ),
                "sentiment_std": (
                    sentiment.std()
                    if len(sentiment) > 1
                    else 0.0
                ),
                "news_count": len(
                    sentiment
                ),
                "positive_ratio": (
                    eligible[
                        "positive_probability"
                    ] >= 0.5
                ).mean(),
                "negative_ratio": (
                    eligible[
                        "negative_probability"
                    ] >= 0.5
                ).mean(),
            }
        )

    return pd.DataFrame(rows)


def attach_news_features(
    market: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:

    news_features = aggregate_news(
        market,
        news,
    )

    return market.merge(
        news_features,
        on=[
            "asset",
            "exchange",
            "timestamp",
        ],
        how="left",
    )


def print_header(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def describe_by_target(
    df: pd.DataFrame,
    asset: str,
) -> None:

    asset_df = df[
        df["asset"] == asset
    ].copy()

    supported = asset_df[
        asset_df["news_count"].notna()
        & asset_df["target_class"].notna()
    ].copy()

    print()
    print(
        f"ASSET: {asset}"
    )

    print("-" * 80)

    print(
        f"News-supported rows: "
        f"{len(supported):,}"
    )

    if supported.empty:
        print(
            "No news-supported observations."
        )
        return

    print()
    print(
        "NEWS FEATURES BY TARGET CLASS"
    )

    print("-" * 80)

    summary = (
        supported
        .groupby("target_class")[
            NEWS_FEATURES
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
    )

    print(
        summary.to_string()
    )


def print_news_density(
    df: pd.DataFrame,
    asset: str,
) -> None:

    asset_df = df[
        df["asset"] == asset
    ].copy()

    supported = asset_df[
        asset_df["news_count"].notna()
    ].copy()

    print()
    print(
        "NEWS DENSITY"
    )

    print("-" * 80)

    if supported.empty:
        print(
            "No news-supported observations."
        )
        return

    print(
        supported[
            "news_count"
        ].describe().to_string()
    )

    print()

    print(
        "NEWS COUNT QUANTILES"
    )

    print("-" * 80)

    quantiles = (
        supported[
            "news_count"
        ]
        .quantile(
            [
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
    )

    print(
        quantiles.to_string()
    )


def print_target_by_news_density(
    df: pd.DataFrame,
    asset: str,
) -> None:

    asset_df = df[
        df["asset"] == asset
    ].copy()

    supported = asset_df[
        asset_df["news_count"].notna()
        & asset_df["target_class"].notna()
    ].copy()

    if supported.empty:
        return

    try:
        supported["news_density_bin"] = pd.qcut(
            supported["news_count"],
            q=4,
            duplicates="drop",
        )
    except ValueError:
        return

    print()
    print(
        "TARGET DISTRIBUTION BY NEWS DENSITY"
    )

    print("-" * 80)

    table = pd.crosstab(
        supported["news_density_bin"],
        supported["target_class"],
        normalize="index",
    )

    table = table.reindex(
        columns=LABELS,
        fill_value=0.0,
    )

    print(
        table.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


def print_source_composition(
    news: pd.DataFrame,
    asset: str,
) -> None:

    asset_news = news[
        news["asset"] == asset
    ].copy()

    print()
    print(
        "NEWS SOURCE COMPOSITION"
    )

    print("-" * 80)

    if asset_news.empty:
        print(
            "No news available."
        )
        return

    if "source" not in asset_news.columns:
        print(
            "Source column unavailable."
        )
        return

    counts = (
        asset_news["source"]
        .value_counts()
    )

    percentages = (
        counts / counts.sum()
    )

    table = pd.DataFrame(
        {
            "articles": counts,
            "share": percentages,
        }
    )

    print(
        table.to_string(
            formatters={
                "share": lambda x: f"{x:.2%}"
            }
        )
    )


def print_daily_coverage(
    df: pd.DataFrame,
    asset: str,
) -> None:

    asset_df = df[
        df["asset"] == asset
    ].copy()

    asset_df["date"] = (
        asset_df["timestamp"]
        .dt.date
    )

    daily = (
        asset_df
        .groupby("date")
        .agg(
            market_rows=(
                "timestamp",
                "size",
            ),
            news_supported_rows=(
                "news_count",
                lambda x: x.notna().sum(),
            ),
            mean_news_count=(
                "news_count",
                "mean",
            ),
        )
    )

    daily["coverage"] = (
        daily["news_supported_rows"]
        / daily["market_rows"]
    )

    print()
    print(
        "DAILY NEWS COVERAGE"
    )

    print("-" * 80)

    print(
        daily.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


def print_sentiment_direction(
    df: pd.DataFrame,
    asset: str,
) -> None:

    asset_df = df[
        (df["asset"] == asset)
        & df["target_class"].notna()
        & df["news_count"].notna()
    ].copy()

    if asset_df.empty:
        return

    print()
    print(
        "SENTIMENT DIRECTION BY TARGET"
    )

    print("-" * 80)

    result = (
        asset_df
        .groupby("target_class")
        .agg(
            sentiment_mean=(
                "sentiment_mean",
                "mean",
            ),
            positive_ratio=(
                "positive_ratio",
                "mean",
            ),
            negative_ratio=(
                "negative_ratio",
                "mean",
            ),
            news_count=(
                "news_count",
                "mean",
            ),
        )
        .reindex(LABELS)
    )

    print(
        result.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


def main():

    print("=" * 80)
    print(
        "PHASE 2.5 — NEWS SIGNAL QUALITY"
    )
    print("=" * 80)

    market = load_market()
    news = load_news()

    print()
    print(
        f"Market input: {MARKET_PATH}"
    )

    print(
        f"News input:   {NEWS_PATH}"
    )

    print(
        f"Market rows:  {len(market):,}"
    )

    print(
        f"News rows:    {len(news):,}"
    )

    print(
        f"News window:  {NEWS_WINDOW}"
    )

    print(
        f"Target horizon: {HORIZON}"
    )

    print(
        f"Target threshold: "
        f"±{TARGET_THRESHOLD:.6%}"
    )

    print(
        f"Locked test starts: "
        f"{LOCKED_START}"
    )

    data = build_target(
        market
    )

    # Development only.
    development = data[
        data["timestamp"]
        < LOCKED_START
    ].copy()

    development = development[
        development["timestamp"]
        >= TRAIN_END
    ].copy()

    development = development[
        development["target_class"]
        .notna()
    ].copy()

    print_header(
        "DEVELOPMENT DATA"
    )

    print(
        f"Development rows: "
        f"{len(development):,}"
    )

    print(
        f"Development period: "
        f"{development['timestamp'].min()} "
        f"→ "
        f"{development['timestamp'].max()}"
    )

    print(
        "Locked holdout excluded: YES"
    )

    print_header(
        "BUILDING NEWS FEATURES"
    )

    result = attach_news_features(
        development,
        news,
    )

    for asset in sorted(
        result["asset"].unique()
    ):

        describe_by_target(
            result,
            asset,
        )

        print_news_density(
            result,
            asset,
        )

        print_target_by_news_density(
            result,
            asset,
        )

        print_source_composition(
            news,
            asset,
        )

        print_daily_coverage(
            result,
            asset,
        )

        print_sentiment_direction(
            result,
            asset,
        )

    print_header(
        "OVERALL NEWS COVERAGE"
    )

    coverage = (
        result["news_count"]
        .notna()
        .mean()
    )

    print(
        f"Development news coverage: "
        f"{coverage:.2%}"
    )

    print()

    coverage_by_asset = (
        result
        .groupby("asset")[
            "news_count"
        ]
        .apply(
            lambda x: x.notna().mean()
        )
    )

    print(
        coverage_by_asset.to_string(
            float_format=lambda x: f"{x:.2%}"
        )
    )

    print_header(
        "PHASE 2.5 DIAGNOSTIC COMPLETE"
    )

    print(
        "Locked observations were not used."
    )

    print(
        "No model fitting or threshold "
        "optimization was performed."
    )


if __name__ == "__main__":
    main()