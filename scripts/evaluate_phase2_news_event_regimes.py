from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.features.market_features import (
    add_normalized_market_features,
)
from src.features.horizon_targets import (
    add_horizon_target,
)


# ============================================================
# CONFIGURATION
# ============================================================

MARKET_INPUT = Path(
    "data/raw/market/phase1_research_market_15m.csv"
)

NEWS_INPUT = Path(
    "data/processed/research_news_sentiment.csv"
)

HORIZON = pd.Timedelta(hours=1)

TARGET_THRESHOLD = 0.00203666

NEWS_WINDOW = pd.Timedelta(hours=1)

TRAIN_START = pd.Timestamp(
    "2026-07-05 00:00:00",
    tz="UTC",
)

TRAIN_END = pd.Timestamp(
    "2026-07-25 00:00:00",
    tz="UTC",
)

OOS_START = pd.Timestamp(
    "2026-07-25 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2026-08-10 00:00:00",
    tz="UTC",
)

LOCKED_START = pd.Timestamp(
    "2026-08-10 00:00:00",
    tz="UTC",
)

ASSETS = [
    "RELIANCE",
    "TCS",
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

EVENT_FEATURES = [
    "news_count",
    "sentiment_magnitude",
    "sentiment_std",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_market() -> pd.DataFrame:

    if not MARKET_INPUT.exists():
        raise FileNotFoundError(
            f"Missing market input: {MARKET_INPUT}"
        )

    df = pd.read_csv(MARKET_INPUT)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    duplicates = df.duplicated(
        ["asset", "exchange", "timestamp"]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Duplicate market rows: {duplicates}"
        )

    return (
        df.sort_values(
            ["asset", "exchange", "timestamp"]
        )
        .reset_index(drop=True)
    )


def load_news() -> pd.DataFrame:

    if not NEWS_INPUT.exists():
        raise FileNotFoundError(
            f"Missing news input: {NEWS_INPUT}"
        )

    df = pd.read_csv(NEWS_INPUT)

    timestamp_candidates = [
        "published_at",
        "timestamp",
        "publication_timestamp",
    ]

    timestamp_column = None

    for column in timestamp_candidates:
        if column in df.columns:
            timestamp_column = column
            break

    if timestamp_column is None:
        raise ValueError(
            "Could not find news timestamp column."
        )

    df["published_at"] = pd.to_datetime(
        df[timestamp_column],
        utc=True,
    )

    if "asset" not in df.columns:
        raise ValueError(
            "News dataset requires an asset column."
        )

    return (
        df.sort_values(
            ["asset", "published_at"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# TARGET
# ============================================================

def build_target(
    market: pd.DataFrame,
) -> pd.DataFrame:

    result = add_normalized_market_features(
        market.copy()
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


# ============================================================
# NEWS FEATURES
# ============================================================

def aggregate_news(
    predictions: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:

    output = predictions[
        ["asset", "timestamp"]
    ].copy()

    feature_rows = []

    for row in output.itertuples(index=False):

        asset = row.asset
        timestamp = row.timestamp

        asset_news = news[
            news["asset"] == asset
        ]

        start = timestamp - NEWS_WINDOW

        window = asset_news[
            (asset_news["published_at"] > start)
            & (asset_news["published_at"] <= timestamp)
        ]

        if window.empty:
            feature_rows.append(
                {
                    "sentiment_mean": np.nan,
                    "sentiment_std": np.nan,
                    "news_count": np.nan,
                    "positive_ratio": np.nan,
                    "negative_ratio": np.nan,
                }
            )
            continue

        sentiments = pd.to_numeric(
            window["sentiment_score"],
            errors="coerce",
        ).dropna()

        if sentiments.empty:

            sentiment_mean = np.nan
            sentiment_std = np.nan
            positive_ratio = np.nan
            negative_ratio = np.nan

        else:

            sentiment_mean = float(
                sentiments.mean()
            )

            sentiment_std = float(
                sentiments.std(ddof=0)
                if len(sentiments) > 1
                else 0.0
            )

            positive_ratio = float(
                (sentiments > 0.05).mean()
            )

            negative_ratio = float(
                (sentiments < -0.05).mean()
            )

        feature_rows.append(
            {
                "sentiment_mean": sentiment_mean,
                "sentiment_std": sentiment_std,
                "news_count": float(len(window)),
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
            }
        )

    features = pd.DataFrame(
        feature_rows,
        index=output.index,
    )

    return pd.concat(
        [output, features],
        axis=1,
    )


# ============================================================
# MODEL
# ============================================================

def fit_model(
    train: pd.DataFrame,
    features: list[str],
):

    usable = train[
        features + ["target_class"]
    ].dropna()

    if usable.empty:
        raise RuntimeError(
            "No usable training observations."
        )

    if usable["target_class"].nunique() < 3:
        raise RuntimeError(
            "Training data must contain all "
            "three target classes."
        )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        usable[features]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X,
        usable["target_class"],
    )

    return scaler, model


def predict(
    scaler,
    model,
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    usable = df[
        features + ["target_class"]
    ].dropna()

    result = df.loc[
        usable.index
    ].copy()

    result["prediction"] = model.predict(
        scaler.transform(
            usable[features]
        )
    )

    return result


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:

    return {
        "accuracy": accuracy_score(
            actual,
            predicted,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            actual,
            predicted,
        ),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=["DOWN", "NEUTRAL", "UP"],
            average="macro",
            zero_division=0,
        ),
        "down_recall": recall_score(
            actual,
            predicted,
            labels=["DOWN"],
            average="macro",
            zero_division=0,
        ),
        "neutral_recall": recall_score(
            actual,
            predicted,
            labels=["NEUTRAL"],
            average="macro",
            zero_division=0,
        ),
        "up_recall": recall_score(
            actual,
            predicted,
            labels=["UP"],
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# EVENT THRESHOLDS
# ============================================================

def calculate_event_thresholds(
    train: pd.DataFrame,
) -> dict[str, float]:

    thresholds = {}

    values = pd.to_numeric(
        train["news_count"],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise RuntimeError(
            "No training values for news_count."
        )

    thresholds["news_count"] = float(
        values.quantile(0.75)
    )

    values = pd.to_numeric(
        train["sentiment_magnitude"],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise RuntimeError(
            "No training values for sentiment_magnitude."
        )

    thresholds["sentiment_magnitude"] = float(
        values.quantile(0.75)
    )

    values = pd.to_numeric(
        train["sentiment_std"],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise RuntimeError(
            "No training values for sentiment_std."
        )

    thresholds["sentiment_std"] = float(
        values.quantile(0.75)
    )

    return thresholds


def add_event_features(
    df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:

    result = df.copy()

    result["high_news_volume"] = (
        result["news_count"]
        >= thresholds["news_count"]
    ).astype(int)

    result["high_sentiment_magnitude"] = (
        result["sentiment_magnitude"]
        >= thresholds["sentiment_magnitude"]
    ).astype(int)

    result["high_sentiment_dispersion"] = (
        result["sentiment_std"]
        >= thresholds["sentiment_std"]
    ).astype(int)

    return result


# ============================================================
# EVENT EXPERIMENT
# ============================================================

def evaluate_event_regime(
    asset: str,
    regime_name: str,
    train: pd.DataFrame,
    oos: pd.DataFrame,
    event_column: str,
) -> None:

    train_event = train[
        train[event_column] == 1
    ].copy()

    oos_event = oos[
        oos[event_column] == 1
    ].copy()

    print()
    print("-" * 80)
    print(f"REGIME: {regime_name}")
    print("-" * 80)

    print(
        f"Training event rows: {len(train_event)}"
    )

    print(
        f"OOS event rows:      {len(oos_event)}"
    )

    if train_event.empty or oos_event.empty:
        print(
            "SKIPPED: insufficient event observations."
        )
        return

    market_train = train_event[
        MARKET_FEATURES + ["target_class"]
    ].dropna()

    market_oos = oos_event[
        MARKET_FEATURES + ["target_class"]
    ].dropna()

    news_train = train_event[
        MARKET_FEATURES
        + NEWS_FEATURES
        + ["target_class"]
    ].dropna()

    news_oos = oos_event[
        MARKET_FEATURES
        + NEWS_FEATURES
        + ["target_class"]
    ].dropna()

    common_train = market_train.index.intersection(
        news_train.index
    )

    common_oos = market_oos.index.intersection(
        news_oos.index
    )

    market_train = market_train.loc[
        common_train
    ]

    news_train = news_train.loc[
        common_train
    ]

    market_oos = market_oos.loc[
        common_oos
    ]

    news_oos = news_oos.loc[
        common_oos
    ]

    if (
        len(market_train) < 20
        or len(market_oos) < 10
    ):
        print(
            "SKIPPED: insufficient matched "
            "event observations."
        )
        return

    market_scaler, market_model = fit_model(
        market_train,
        MARKET_FEATURES,
    )

    news_scaler, news_model = fit_model(
        news_train,
        MARKET_FEATURES + NEWS_FEATURES,
    )

    market_predictions = predict(
        market_scaler,
        market_model,
        market_oos,
        MARKET_FEATURES,
    )

    news_predictions = predict(
        news_scaler,
        news_model,
        news_oos,
        MARKET_FEATURES + NEWS_FEATURES,
    )

    common_oos = market_predictions.index.intersection(
        news_predictions.index
    )

    market_predictions = market_predictions.loc[
        common_oos
    ]

    news_predictions = news_predictions.loc[
        common_oos
    ]

    actual = market_predictions[
        "target_class"
    ]

    market_metrics = calculate_metrics(
        actual,
        market_predictions["prediction"],
    )

    news_metrics = calculate_metrics(
        actual,
        news_predictions["prediction"],
    )

    print()
    print(
        f"Matched training rows: "
        f"{len(common_train)}"
    )

    print(
        f"Matched OOS rows:      "
        f"{len(common_oos)}"
    )

    print()
    print(
        f"{'Metric':<22}"
        f"{'Market':>12}"
        f"{'Market+News':>15}"
        f"{'Delta':>12}"
    )

    print("-" * 61)

    metric_names = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("down_recall", "DOWN Recall"),
        ("neutral_recall", "NEUTRAL Recall"),
        ("up_recall", "UP Recall"),
    ]

    for key, label in metric_names:

        market_value = market_metrics[key]
        news_value = news_metrics[key]

        print(
            f"{label:<22}"
            f"{market_value:>12.4f}"
            f"{news_value:>15.4f}"
            f"{news_value - market_value:>12.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print(
        "PHASE 2.7 SELECTIVE NEWS SIGNAL / "
        "EVENT REGIME ANALYSIS"
    )
    print("=" * 80)

    market = load_market()

    news = load_news()

    prepared = build_target(
        market
    )

    development = prepared[
        (prepared["timestamp"] >= TRAIN_START)
        & (prepared["timestamp"] < LOCKED_START)
    ].copy()

    print()
    print("DEVELOPMENT DATA")
    print("-" * 80)

    print(
        f"Development rows: {len(development)}"
    )

    print(
        f"Development period: "
        f"{TRAIN_START} ? {LOCKED_START}"
    )

    locked_rows = (
        prepared["timestamp"]
        >= LOCKED_START
    ).sum()

    print(
        f"Locked rows: {locked_rows}"
    )

    for asset in ASSETS:

        print()
        print("=" * 80)
        print(f"ASSET: {asset}")
        print("=" * 80)

        asset_market = development[
            development["asset"] == asset
        ].copy()

        asset_news = news[
            news["asset"] == asset
        ].copy()

        news_features = aggregate_news(
            asset_market,
            asset_news,
        )

        merged = asset_market.merge(
            news_features,
            on=["asset", "timestamp"],
            how="left",
        )

        merged["sentiment_magnitude"] = (
            pd.to_numeric(
                merged["sentiment_mean"],
                errors="coerce",
            ).abs()
        )

        train = merged[
            (merged["timestamp"] >= TRAIN_START)
            & (merged["timestamp"] < TRAIN_END)
        ].copy()

        oos = merged[
            (merged["timestamp"] >= OOS_START)
            & (merged["timestamp"] < OOS_END)
        ].copy()

        train = train[
            train["target_class"].notna()
        ]

        oos = oos[
            oos["target_class"].notna()
        ]

        thresholds = calculate_event_thresholds(
            train
        )

        train = add_event_features(
            train,
            thresholds,
        )

        oos = add_event_features(
            oos,
            thresholds,
        )

        print()
        print("FROZEN EVENT THRESHOLDS")
        print("-" * 80)

        for name, value in thresholds.items():

            print(
                f"{name}: {value:.6f}"
            )

        print()
        print("EVENT COVERAGE")
        print("-" * 80)

        coverage_columns = [
            (
                "high_news_volume",
                "High news volume",
            ),
            (
                "high_sentiment_magnitude",
                "High sentiment magnitude",
            ),
            (
                "high_sentiment_dispersion",
                "High sentiment dispersion",
            ),
        ]

        for column, label in coverage_columns:

            print(
                f"{label:<30}"
                f"Train: {train[column].mean():.2%}  "
                f"OOS: {oos[column].mean():.2%}"
            )

        regimes = [
            (
                "HIGH NEWS VOLUME",
                "high_news_volume",
            ),
            (
                "HIGH SENTIMENT MAGNITUDE",
                "high_sentiment_magnitude",
            ),
            (
                "HIGH SENTIMENT DISPERSION",
                "high_sentiment_dispersion",
            ),
        ]

        for regime_name, event_column in regimes:

            evaluate_event_regime(
                asset,
                regime_name,
                train,
                oos,
                event_column,
            )

    print()
    print("=" * 80)
    print("LOCKED HOLDOUT PROTECTION")
    print("=" * 80)

    print(
        "Locked observations were excluded "
        "from threshold calculation, "
        "model fitting, and evaluation."
    )

    print()
    print(
        "PHASE 2.7 EVENT REGIME EXPERIMENT COMPLETE"
    )


if __name__ == "__main__":
    main()
