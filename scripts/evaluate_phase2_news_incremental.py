from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from src.features.market_features import (
    add_normalized_market_features,
)
from src.features.horizon_targets import (
    add_horizon_target,
)


# ============================================================
# PATHS
# ============================================================

MARKET_PATH = Path(
    "data/raw/market/phase1_research_market_15m.csv"
)

NEWS_PATH = Path(
    "data/processed/research_news_sentiment.csv"
)


# ============================================================
# EXPERIMENT WINDOWS
# ============================================================

NEWS_TRAIN_START = pd.Timestamp(
    "2026-07-05 00:00:00",
    tz="UTC",
)

NEWS_OOS_START = pd.Timestamp(
    "2026-07-25 00:00:00",
    tz="UTC",
)

LOCKED_START = pd.Timestamp(
    "2026-08-10 00:00:00",
    tz="UTC",
)


# ============================================================
# TARGET DEFINITION
# ============================================================

HORIZON = pd.Timedelta(hours=1)

TARGET_THRESHOLD = 0.00203666


# ============================================================
# FEATURES
# ============================================================

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

LABELS = [
    "DOWN",
    "NEUTRAL",
    "UP",
]


# ============================================================
# MARKET LOADING
# ============================================================

def load_market() -> pd.DataFrame:

    if not MARKET_PATH.exists():
        raise FileNotFoundError(
            f"Market file not found: {MARKET_PATH}"
        )

    df = pd.read_csv(
        MARKET_PATH
    )

    required = {
        "asset",
        "exchange",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required market columns: "
            f"{sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    duplicate_count = df.duplicated(
        [
            "asset",
            "exchange",
            "timestamp",
        ]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            f"market observations."
        )

    return (
        df.sort_values(
            [
                "asset",
                "exchange",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# NEWS LOADING
# ============================================================

def load_news() -> pd.DataFrame:

    if not NEWS_PATH.exists():
        raise FileNotFoundError(
            f"News sentiment file not found: "
            f"{NEWS_PATH}"
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

    missing = (
        required
        - set(news.columns)
    )

    if missing:
        raise ValueError(
            "Missing required news columns: "
            f"{sorted(missing)}"
        )

    news["published_at"] = pd.to_datetime(
        news["published_at"],
        utc=True,
    )

    news = (
        news.sort_values(
            [
                "asset",
                "published_at",
            ]
        )
        .reset_index(drop=True)
    )

    return news


# ============================================================
# TARGET
# ============================================================

def build_target(
    market: pd.DataFrame,
) -> pd.DataFrame:

    result = add_normalized_market_features(
        market
    )

    result = add_horizon_target(
        result,
        HORIZON,
    )

    returns = result[
        "target_return"
    ]

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

def aggregate_news_for_predictions(
    predictions: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate only news published during the trailing
    one-hour information window.

    For a prediction at time T, only articles satisfying:

        T - 1 hour <= published_at <= T

    are eligible.

    No future information is used.
    """

    rows = []

    for _, prediction in predictions.iterrows():

        asset = prediction["asset"]
        timestamp = prediction["timestamp"]

        window_start = (
            timestamp
            - pd.Timedelta(hours=1)
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

                "news_count": (
                    len(sentiment)
                ),

                "positive_ratio": (
                    (
                        eligible[
                            "positive_probability"
                        ]
                        >= 0.5
                    ).mean()
                ),

                "negative_ratio": (
                    (
                        eligible[
                            "negative_probability"
                        ]
                        >= 0.5
                    ).mean()
                ),
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "asset",
                "exchange",
                "timestamp",
                *NEWS_FEATURES,
            ]
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ADD NEWS FEATURES
# ============================================================

def attach_news_features(
    predictions: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:

    news_features = (
        aggregate_news_for_predictions(
            predictions,
            news,
        )
    )

    result = predictions.merge(
        news_features,
        on=[
            "asset",
            "exchange",
            "timestamp",
        ],
        how="left",
    )

    return result


# ============================================================
# MODEL EVALUATION
# ============================================================

def fit_and_evaluate(
    train: pd.DataFrame,
    oos: pd.DataFrame,
    features: list[str],
) -> dict:

    train_usable = train[
        features + ["target_class"]
    ].dropna()

    oos_usable = oos[
        features + ["target_class"]
    ].dropna()

    if train_usable.empty:
        raise RuntimeError(
            "No usable training rows."
        )

    if oos_usable.empty:
        raise RuntimeError(
            "No usable OOS rows."
        )

    if train_usable[
        "target_class"
    ].nunique() < 3:

        raise RuntimeError(
            "Training target does not contain "
            "all three classes."
        )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_usable[features]
    )

    X_oos = scaler.transform(
        oos_usable[features]
    )

    y_train = train_usable[
        "target_class"
    ]

    y_oos = oos_usable[
        "target_class"
    ]

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_oos
    )

    return {
        "training_rows": len(
            train_usable
        ),

        "oos_rows": len(
            oos_usable
        ),

        "accuracy": accuracy_score(
            y_oos,
            predictions,
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                y_oos,
                predictions,
            )
        ),

        "macro_f1": f1_score(
            y_oos,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),

        "oos_index": oos_usable.index,
    }


# ============================================================
# RESULT PRINTING
# ============================================================

def print_result(
    name: str,
    result: dict,
) -> None:

    print()
    print(name)
    print("-" * 80)

    print(
        f"Training rows:       "
        f"{result['training_rows']:,}"
    )

    print(
        f"OOS rows:            "
        f"{result['oos_rows']:,}"
    )

    print(
        f"Accuracy:            "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy:   "
        f"{result['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro-F1:            "
        f"{result['macro_f1']:.4f}"
    )


# ============================================================
# COVERAGE
# ============================================================

def calculate_news_coverage(
    df: pd.DataFrame,
) -> float:

    return (
        df[
            NEWS_FEATURES
        ]
        .notna()
        .all(axis=1)
        .mean()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "PHASE 2.3A — NEWS INCREMENTAL SIGNAL"
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
        f"Market rows: {len(market):,}"
    )

    print(
        f"News rows:   {len(news):,}"
    )

    print(
        f"Horizon: {HORIZON}"
    )

    print(
        f"Frozen target threshold: "
        f"±{TARGET_THRESHOLD:.6%}"
    )

    print(
        f"News-supported training start: "
        f"{NEWS_TRAIN_START}"
    )

    print(
        f"OOS starts: "
        f"{NEWS_OOS_START}"
    )

    print(
        f"Locked test starts: "
        f"{LOCKED_START}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    data = build_target(
        market
    )

    # --------------------------------------------------------
    # DEVELOPMENT / LOCKED SPLIT
    # --------------------------------------------------------

    development = data[
        data["timestamp"]
        < LOCKED_START
    ].copy()

    development = development[
        development["target_class"]
        .notna()
    ].copy()

    locked = data[
        data["timestamp"]
        >= LOCKED_START
    ].copy()

    # --------------------------------------------------------
    # NEWS-SUPPORTED CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    training = development[
        (
            development["timestamp"]
            >= NEWS_TRAIN_START
        )
        & (
            development["timestamp"]
            < NEWS_OOS_START
        )
    ].copy()

    oos = development[
        (
            development["timestamp"]
            >= NEWS_OOS_START
        )
        & (
            development["timestamp"]
            < LOCKED_START
        )
    ].copy()

    print()
    print("=" * 80)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 80)

    print(
        f"Training period: "
        f"{NEWS_TRAIN_START} → "
        f"{NEWS_OOS_START}"
    )

    print(
        f"OOS period:      "
        f"{NEWS_OOS_START} → "
        f"{LOCKED_START}"
    )

    print(
        f"Locked period:    "
        f">= {LOCKED_START}"
    )

    print()
    print(
        f"Training rows: {len(training):,}"
    )

    print(
        f"OOS rows:      {len(oos):,}"
    )

    print(
        f"Locked rows:   {len(locked):,}"
    )

    # --------------------------------------------------------
    # NEWS FEATURES
    # --------------------------------------------------------

    print()
    print(
        "Building news features using only "
        "the trailing 1-hour information window..."
    )

    training = attach_news_features(
        training,
        news,
    )

    oos = attach_news_features(
        oos,
        news,
    )

    # --------------------------------------------------------
    # ASSET EVALUATION
    # --------------------------------------------------------

    for asset in sorted(
        development["asset"].unique()
    ):

        print()
        print("=" * 80)
        print(
            f"ASSET: {asset}"
        )
        print("=" * 80)

        asset_training = training[
            training["asset"] == asset
        ].copy()

        asset_oos = oos[
            oos["asset"] == asset
        ].copy()

        print()
        print(
            f"Training rows: "
            f"{len(asset_training):,}"
        )

        print(
            f"OOS rows: "
            f"{len(asset_oos):,}"
        )

        # ----------------------------------------------------
        # NEWS COVERAGE
        # ----------------------------------------------------

        training_coverage = (
            calculate_news_coverage(
                asset_training
            )
        )

        oos_coverage = (
            calculate_news_coverage(
                asset_oos
            )
        )

        print()
        print(
            "NEWS FEATURE COVERAGE"
        )
        print("-" * 80)

        print(
            f"Training: "
            f"{training_coverage:.2%}"
        )

        print(
            f"OOS:      "
            f"{oos_coverage:.2%}"
        )

        # ----------------------------------------------------
        # MARKET-ONLY
        # ----------------------------------------------------

        market_result = fit_and_evaluate(
            asset_training,
            asset_oos,
            MARKET_FEATURES,
        )

        print_result(
            "MARKET-ONLY",
            market_result,
        )

        # ----------------------------------------------------
        # COMMON NEWS-SUPPORTED SAMPLE
        # ----------------------------------------------------

        common_training = asset_training[
            asset_training[
                NEWS_FEATURES
            ]
            .notna()
            .all(axis=1)
        ].copy()

        common_oos = asset_oos[
            asset_oos[
                NEWS_FEATURES
            ]
            .notna()
            .all(axis=1)
        ].copy()

        if common_training.empty:

            raise RuntimeError(
                f"No news-supported training rows "
                f"for asset {asset}."
            )

        if common_oos.empty:

            raise RuntimeError(
                f"No news-supported OOS rows "
                f"for asset {asset}."
            )

        # ----------------------------------------------------
        # MARKET + NEWS
        # ----------------------------------------------------

        news_result = fit_and_evaluate(
            common_training,
            common_oos,
            MARKET_FEATURES
            + NEWS_FEATURES,
        )

        print_result(
            "MARKET + NEWS",
            news_result,
        )

        # ----------------------------------------------------
        # INCREMENTAL CHANGE
        # ----------------------------------------------------

        print()
        print(
            "INCREMENTAL CHANGE"
        )
        print("-" * 80)

        print(
            f"Accuracy: "
            f"{news_result['accuracy'] - market_result['accuracy']:+.4f}"
            )

        print(
            f"Balanced accuracy: "
            f"{news_result['balanced_accuracy'] - market_result['balanced_accuracy']:+.4f}"
            )

        print(
            f"Macro-F1: "
            f"{news_result['macro_f1'] - market_result['macro_f1']:+.4f}"
            )

    # --------------------------------------------------------
    # LOCKED HOLDOUT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "LOCKED HOLDOUT PROTECTION"
    )
    print("=" * 80)

    print(
        f"Locked rows available: "
        f"{len(locked):,}"
    )

    print(
        "Locked observations were not used "
        "for training, news-feature selection, "
        "model fitting, or metric comparison."
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "PHASE 2.3A NEWS INCREMENTAL "
        "EXPERIMENT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()