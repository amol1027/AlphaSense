from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
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

EVENT_FEATURE = "sentiment_magnitude"


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
# NEWS AGGREGATION
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
# DIAGNOSTICS
# ============================================================

def print_class_distribution(
    name: str,
    df: pd.DataFrame,
) -> None:

    counts = (
        df["target_class"]
        .value_counts()
        .reindex(
            ["DOWN", "NEUTRAL", "UP"],
            fill_value=0,
        )
    )

    total = len(df)

    print()
    print(name)
    print("-" * 60)

    for label in [
        "DOWN",
        "NEUTRAL",
        "UP",
    ]:

        count = int(counts[label])

        percentage = (
            count / total
            if total
            else 0.0
        )

        print(
            f"{label:<10}"
            f"{count:>6}"
            f"  ({percentage:>7.2%})"
        )

    print(
        f"{'TOTAL':<10}"
        f"{total:>6}"
    )


def print_confusion_matrix(
    name: str,
    actual: pd.Series,
    predicted: pd.Series,
) -> None:

    labels = [
        "DOWN",
        "NEUTRAL",
        "UP",
    ]

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels,
    )

    print()
    print(name)
    print("-" * 60)

    print(
        f"{'':>12}"
        f"{'DOWN':>10}"
        f"{'NEUTRAL':>12}"
        f"{'UP':>10}"
    )

    for i, label in enumerate(labels):

        print(
            f"{label:>12}"
            f"{matrix[i, 0]:>10}"
            f"{matrix[i, 1]:>12}"
            f"{matrix[i, 2]:>10}"
        )


def print_metrics(
    name: str,
    actual: pd.Series,
    predicted: pd.Series,
) -> None:

    print()
    print(name)
    print("-" * 60)

    print(
        f"Accuracy:          "
        f"{accuracy_score(actual, predicted):.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_score(actual, predicted):.4f}"
    )

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=["DOWN", "NEUTRAL", "UP"],
        average="macro",
        zero_division=0,
    )

    print(
        f"Macro-F1:           "
        f"{macro_f1:.4f}"
    )

    for label in [
        "DOWN",
        "NEUTRAL",
        "UP",
    ]:

        recall = recall_score(
            actual,
            predicted,
            labels=[label],
            average="macro",
            zero_division=0,
        )

        print(
            f"{label} Recall:       "
            f"{recall:.4f}"
        )


# ============================================================
# ASSET AUDIT
# ============================================================

def audit_asset(
    asset: str,
    development: pd.DataFrame,
    news: pd.DataFrame,
) -> None:

    print()
    print("=" * 80)
    print(f"ASSET AUDIT: {asset}")
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

    # --------------------------------------------------------
    # THRESHOLD AUDIT
    # --------------------------------------------------------

    training_values = pd.to_numeric(
        train[EVENT_FEATURE],
        errors="coerce",
    ).dropna()

    threshold = float(
        training_values.quantile(0.75)
    )

    print()
    print("THRESHOLD AUDIT")
    print("-" * 60)

    print(
        f"Training period: "
        f"{TRAIN_START} → {TRAIN_END}"
    )

    print(
        f"Threshold feature: "
        f"{EVENT_FEATURE}"
    )

    print(
        f"Training non-null values: "
        f"{len(training_values)}"
    )

    print(
        f"P75 threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Training min: "
        f"{training_values.min():.6f}"
    )

    print(
        f"Training max: "
        f"{training_values.max():.6f}"
    )

    # --------------------------------------------------------
    # EVENT FLAG
    # --------------------------------------------------------

    train["event"] = (
        train[EVENT_FEATURE]
        >= threshold
    ).astype(int)

    oos["event"] = (
        oos[EVENT_FEATURE]
        >= threshold
    ).astype(int)

    train_event = train[
        train["event"] == 1
    ].copy()

    oos_event = oos[
        oos["event"] == 1
    ].copy()

    print()
    print("EVENT SAMPLE")
    print("-" * 60)

    print(
        f"Training event rows: "
        f"{len(train_event)}"
    )

    print(
        f"OOS event rows: "
        f"{len(oos_event)}"
    )

    print(
        f"Training coverage: "
        f"{train['event'].mean():.2%}"
    )

    print(
        f"OOS coverage: "
        f"{oos['event'].mean():.2%}"
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTIONS
    # --------------------------------------------------------

    print_class_distribution(
        "TRAINING EVENT CLASS DISTRIBUTION",
        train_event,
    )

    print_class_distribution(
        "OOS EVENT CLASS DISTRIBUTION",
        oos_event,
    )

    # --------------------------------------------------------
    # EXACT TIMESTAMPS
    # --------------------------------------------------------

    print()
    print("OOS EVENT TIMESTAMPS")
    print("-" * 60)

    for timestamp in (
        oos_event["timestamp"]
        .sort_values()
        .tolist()
    ):

        print(timestamp)

    # --------------------------------------------------------
    # MATCHED MODEL DATA
    # --------------------------------------------------------

    market_train = train_event[
        MARKET_FEATURES
        + ["target_class"]
    ].dropna()

    news_train = train_event[
        MARKET_FEATURES
        + NEWS_FEATURES
        + ["target_class"]
    ].dropna()

    market_oos = oos_event[
        MARKET_FEATURES
        + ["target_class"]
    ].dropna()

    news_oos = oos_event[
        MARKET_FEATURES
        + NEWS_FEATURES
        + ["target_class"]
    ].dropna()

    common_train = (
        market_train.index
        .intersection(news_train.index)
    )

    common_oos = (
        market_oos.index
        .intersection(news_oos.index)
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

    print()
    print("MATCHED SAMPLE AUDIT")
    print("-" * 60)

    print(
        f"Matched training rows: "
        f"{len(common_train)}"
    )

    print(
        f"Matched OOS rows: "
        f"{len(common_oos)}"
    )

    if (
        len(common_train) < 20
        or len(common_oos) < 10
    ):

        print(
            "AUDIT WARNING: matched sample "
            "is small."
        )

        return

    # --------------------------------------------------------
    # MODEL FIT
    # --------------------------------------------------------

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

    common_predictions = (
        market_predictions.index
        .intersection(
            news_predictions.index
        )
    )

    market_predictions = (
        market_predictions.loc[
            common_predictions
        ]
    )

    news_predictions = (
        news_predictions.loc[
            common_predictions
        ]
    )

    actual = market_predictions[
        "target_class"
    ]

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    print_confusion_matrix(
        "MARKET-ONLY CONFUSION MATRIX",
        actual,
        market_predictions["prediction"],
    )

    print_confusion_matrix(
        "MARKET + NEWS CONFUSION MATRIX",
        actual,
        news_predictions["prediction"],
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    print_metrics(
        "MARKET-ONLY METRICS",
        actual,
        market_predictions["prediction"],
    )

    print_metrics(
        "MARKET + NEWS METRICS",
        actual,
        news_predictions["prediction"],
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    print()
    print("SANITY CHECK")
    print("-" * 60)

    print(
        f"Actual rows: "
        f"{len(actual)}"
    )

    print(
        f"Market prediction rows: "
        f"{len(market_predictions)}"
    )

    print(
        f"News prediction rows: "
        f"{len(news_predictions)}"
    )

    print(
        "Same OOS timestamps: "
        f"{market_predictions.index.equals(news_predictions.index)}"
    )

    locked_in_oos = (
        oos_event["timestamp"]
        >= LOCKED_START
    ).any()

    print(
        f"Locked observations included: "
        f"{locked_in_oos}"
    )

    if locked_in_oos:
        raise RuntimeError(
            "LOCKED HOLDOUT CONTAMINATION DETECTED"
        )

    print()
    print(
        "AUDIT STATUS: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print(
        "PHASE 2.7 HIGH-SENTIMENT-MAGNITUDE AUDIT"
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
    print("DATA BOUNDARIES")
    print("-" * 60)

    print(
        f"Training: "
        f"{TRAIN_START} → {TRAIN_END}"
    )

    print(
        f"OOS: "
        f"{OOS_START} → {OOS_END}"
    )

    print(
        f"Locked: "
        f"{LOCKED_START} onward"
    )

    locked_rows = (
        prepared["timestamp"]
        >= LOCKED_START
    ).sum()

    print(
        f"Locked rows available in dataset: "
        f"{locked_rows}"
    )

    for asset in ASSETS:

        audit_asset(
            asset,
            development,
            news,
        )

    print()
    print("=" * 80)
    print(
        "PHASE 2.7 AUDIT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()