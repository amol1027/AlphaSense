from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
)
from sklearn.preprocessing import StandardScaler

from src.features.market_features import (
    add_normalized_market_features,
)
from src.features.horizon_targets import (
    add_horizon_target,
)


MARKET_INPUT = Path(
    "data/raw/market/phase1_research_market_15m.csv"
)

NEWS_INPUT = Path(
    "data/processed/research_news_sentiment.csv"
)

HORIZON = pd.Timedelta(hours=1)

THRESHOLD = 0.00203666

NEWS_WINDOW = pd.Timedelta(hours=1)

BURST_LOOKBACK = pd.Timedelta(days=1)

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
    "news_count",
    "news_burst",
    "sentiment_std",
    "positive_ratio",
    "negative_ratio",
    "sentiment_imbalance",
    "source_diversity",
]


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

    return df.sort_values(
        ["asset", "published_at"]
    ).reset_index(drop=True)


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
                    "news_count": np.nan,
                    "news_burst": np.nan,
                    "sentiment_std": np.nan,
                    "positive_ratio": np.nan,
                    "negative_ratio": np.nan,
                    "sentiment_imbalance": np.nan,
                    "source_diversity": np.nan,
                }
            )
            continue

        sentiments = pd.to_numeric(
            window["sentiment_score"],
            errors="coerce",
        ).dropna()

        if sentiments.empty:
            sentiment_std = np.nan
            positive_ratio = np.nan
            negative_ratio = np.nan
            sentiment_imbalance = np.nan
        else:
            sentiment_std = (
                sentiments.std(ddof=0)
                if len(sentiments) > 1
                else 0.0
            )

            positive_ratio = (
                sentiments > 0.05
            ).mean()

            negative_ratio = (
                sentiments < -0.05
            ).mean()

            sentiment_imbalance = (
                positive_ratio
                - negative_ratio
            )

        source_diversity = (
            window["source"]
            .nunique()
            if "source" in window.columns
            else np.nan
        )

        feature_rows.append(
            {
                "news_count": float(len(window)),
                "news_burst": np.nan,
                "sentiment_std": sentiment_std,
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "sentiment_imbalance": sentiment_imbalance,
                "source_diversity": (
                    float(source_diversity)
                    if not pd.isna(source_diversity)
                    else np.nan
                ),
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


def add_news_burst(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:

    result = feature_df.copy()

    counts = result["news_count"]

    rolling_mean = (
        counts.rolling(
            window=20,
            min_periods=1,
        )
        .mean()
    )

    result["news_burst"] = (
        counts / rolling_mean.replace(0, np.nan)
    )

    return result


def calculate_event_thresholds(
    train: pd.DataFrame,
) -> dict[str, float]:

    thresholds = {}

    for column in [
        "news_count",
        "news_burst",
    ]:
        values = pd.to_numeric(
            train[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            raise RuntimeError(
                f"No training values for {column}."
            )

        thresholds[column] = float(
            values.quantile(0.75)
        )

    return thresholds


def add_event_regime_features(
    df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:

    result = df.copy()

    news_count = pd.to_numeric(
        result["news_count"],
        errors="coerce",
    ).fillna(0)

    news_burst = pd.to_numeric(
        result["news_burst"],
        errors="coerce",
    ).fillna(0)

    result["high_news_event"] = (
        news_count >= thresholds["news_count"]
    ).astype(int)

    result["high_news_burst"] = (
        news_burst >= thresholds["news_burst"]
    ).astype(int)

    return result


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
        features
    ].dropna()

    result = df.loc[
        usable.index
    ].copy()

    probabilities = model.predict_proba(
        scaler.transform(
            usable[features]
        )
    )

    for index, class_name in enumerate(
        model.classes_
    ):
        result[
            f"prob_{class_name.lower()}"
        ] = probabilities[:, index]

    result["prediction"] = (
        model.predict(
            scaler.transform(
                usable[features]
            )
        )
    )

    return result


def apply_selective_news_policy(
    df: pd.DataFrame,
    up_probability_threshold: float,
    down_probability_threshold: float,
) -> pd.Series:

    result = []

    for row in df.itertuples():

        is_event = (
            getattr(
                row,
                "high_news_event",
                0,
            )
            == 1
        )

        if not is_event:
            result.append("NEUTRAL")
            continue

        prob_up = getattr(
            row,
            "prob_up",
            0.0,
        )

        prob_down = getattr(
            row,
            "prob_down",
            0.0,
        )

        if prob_up >= up_probability_threshold:
            result.append("UP")

        elif (
            prob_down
            >= down_probability_threshold
        ):
            result.append("DOWN")

        else:
            result.append("NEUTRAL")

    return pd.Series(
        result,
        index=df.index,
        dtype="string",
    )


def evaluate(
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
            average="macro",
            zero_division=0,
        ),
        "up_precision": precision_score(
            actual,
            predicted,
            labels=["UP"],
            average="macro",
            zero_division=0,
        ),
        "down_precision": precision_score(
            actual,
            predicted,
            labels=["DOWN"],
            average="macro",
            zero_division=0,
        ),
    }


def evaluate_asset(
    asset: str,
    development: pd.DataFrame,
    news: pd.DataFrame,
) -> None:

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

    news_features = add_news_burst(
        news_features
    )

    merged = asset_market.merge(
        news_features,
        on=["asset", "timestamp"],
        how="left",
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

    print()
    print("=" * 80)
    print(f"ASSET: {asset}")
    print("=" * 80)

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"OOS rows:      {len(oos)}"
    )

    thresholds = calculate_event_thresholds(
        train
    )

    print()
    print("EVENT THRESHOLDS")
    print("-" * 80)

    for name, value in thresholds.items():
        print(
            f"{name}: {value:.6f}"
        )

    train = add_event_regime_features(
        train,
        thresholds,
    )

    oos = add_event_regime_features(
        oos,
        thresholds,
    )

    print()
    print("EVENT COVERAGE")
    print("-" * 80)

    print(
        "Training high-news-event: "
        f"{train['high_news_event'].mean():.2%}"
    )

    print(
        "OOS high-news-event:      "
        f"{oos['high_news_event'].mean():.2%}"
    )

    model_features = (
        MARKET_FEATURES
        + NEWS_FEATURES
    )

    scaler, model = fit_model(
        train,
        model_features,
    )

    predicted_oos = predict(
        scaler,
        model,
        oos,
        model_features,
    )

    selective_prediction = (
        apply_selective_news_policy(
            predicted_oos,
            up_probability_threshold=0.5,
            down_probability_threshold=0.5,
        )
    )

    actual = predicted_oos[
        "target_class"
    ]

    metrics = evaluate(
        actual,
        selective_prediction,
    )

    coverage = (
        selective_prediction
        != "NEUTRAL"
    ).mean()

    print()
    print("SELECTIVE NEWS POLICY")
    print("-" * 80)

    print(
        f"Coverage:             {coverage:.2%}"
    )

    print(
        f"Accuracy:             "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy:    "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro-F1:             "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"UP precision:         "
        f"{metrics['up_precision']:.4f}"
    )

    print(
        f"DOWN precision:       "
        f"{metrics['down_precision']:.4f}"
    )


def main() -> None:

    print("=" * 80)
    print(
        "PHASE 2.7 — SELECTIVE NEWS SIGNAL / "
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
        f"Development rows: "
        f"{len(development)}"
    )

    print(
    "Development period: "
    f"{TRAIN_START} "
    f"→ "
    f"{LOCKED_START}"
)

    locked_rows = (
        prepared["timestamp"]
        >= LOCKED_START
    ).sum()

    print(
        f"Locked rows: {locked_rows}"
    )

    for asset in ASSETS:
        evaluate_asset(
            asset,
            development,
            news,
        )

    print()
    print("=" * 80)
    print(
        "LOCKED HOLDOUT PROTECTION"
    )
    print("=" * 80)

    print(
        "Locked observations were excluded "
        "from feature threshold calculation, "
        "model fitting, and evaluation."
    )

    print()
    print(
        "PHASE 2.7 SELECTIVE NEWS EXPERIMENT COMPLETE"
    )


if __name__ == "__main__":
    main()