from pathlib import Path

import numpy as np
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
# FROZEN EXPERIMENT PARAMETERS
# ============================================================

NEWS_TRAIN_START = pd.Timestamp(
    "2026-07-05 00:00:00",
    tz="UTC",
)

OOS_START = pd.Timestamp(
    "2026-07-25 00:00:00",
    tz="UTC",
)

LOCKED_START = pd.Timestamp(
    "2026-08-10 00:00:00",
    tz="UTC",
)

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
# LOAD MARKET
# ============================================================

def load_market() -> pd.DataFrame:

    if not MARKET_PATH.exists():
        raise FileNotFoundError(
            f"Market file not found: {MARKET_PATH}"
        )

    df = pd.read_csv(MARKET_PATH)

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
# LOAD NEWS
# ============================================================

def load_news() -> pd.DataFrame:

    if not NEWS_PATH.exists():
        raise FileNotFoundError(
            f"News sentiment file not found: {NEWS_PATH}"
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

    return (
        news.sort_values(
            [
                "asset",
                "published_at",
            ]
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

        sentiment = eligible[
            "sentiment_score"
        ]

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
# MERGE NEWS
# ============================================================

def attach_news_features(
    market_data: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:

    news_features = (
        aggregate_news_for_predictions(
            market_data,
            news,
        )
    )

    return market_data.merge(
        news_features,
        on=[
            "asset",
            "exchange",
            "timestamp",
        ],
        how="left",
    )


# ============================================================
# MODEL
# ============================================================

def fit_and_predict(
    training: pd.DataFrame,
    oos: pd.DataFrame,
    features: list[str],
):

    train_usable = training[
        features + ["target_class"]
    ].dropna()

    oos_usable = oos[
        features + ["target_class"]
    ].dropna()

    if train_usable.empty:
        raise RuntimeError(
            "No usable training observations."
        )

    if oos_usable.empty:
        raise RuntimeError(
            "No usable OOS observations."
        )

    y_train = train_usable[
        "target_class"
    ]

    y_oos = oos_usable[
        "target_class"
    ]

    if y_train.nunique() < 3:
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
        "y_true": y_oos.to_numpy(),
        "predictions": predictions,
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
    }


# ============================================================
# PRINT MODEL RESULT
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
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    result: dict,
) -> None:

    matrix = pd.crosstab(
        pd.Series(
            result["y_true"],
            name="actual",
        ),
        pd.Series(
            result["predictions"],
            name="predicted",
        ),
        dropna=False,
    )

    matrix = matrix.reindex(
        index=LABELS,
        columns=LABELS,
        fill_value=0,
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 80)

    print(
        matrix.to_string()
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(
    name: str,
    df: pd.DataFrame,
) -> None:

    distribution = (
        df["target_class"]
        .value_counts(
            normalize=True
        )
        .reindex(LABELS)
        .fillna(0)
    )

    print()
    print(name)
    print("-" * 80)

    print(
        distribution.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "PHASE 2.3B — MATCHED-SAMPLE "
        "NEWS INCREMENTAL SIGNAL"
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
        f"{OOS_START}"
    )

    print(
        f"Locked test starts: "
        f"{LOCKED_START}"
    )

    # --------------------------------------------------------
    # Build target
    # --------------------------------------------------------

    data = build_target(
        market
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    training = data[
        (data["timestamp"] >= NEWS_TRAIN_START)
        & (data["timestamp"] < OOS_START)
        & (data["target_class"].notna())
    ].copy()

    oos = data[
        (data["timestamp"] >= OOS_START)
        & (data["timestamp"] < LOCKED_START)
        & (data["target_class"].notna())
    ].copy()

    locked = data[
        (data["timestamp"] >= LOCKED_START)
    ].copy()

    print()
    print("=" * 80)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 80)

    print(
        f"Training period: "
        f"{NEWS_TRAIN_START} → {OOS_START}"
    )

    print(
        f"OOS period:      "
        f"{OOS_START} → {LOCKED_START}"
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
    # Attach news
    # --------------------------------------------------------

    print()
    print(
        "Building news features using only "
        "the trailing 1-hour information window..."
    )

    training_with_news = attach_news_features(
        training,
        news,
    )

    oos_with_news = attach_news_features(
        oos,
        news,
    )

    # --------------------------------------------------------
    # Asset evaluation
    # --------------------------------------------------------

    for asset in sorted(
        training["asset"].unique()
    ):

        print()
        print("=" * 80)
        print(f"ASSET: {asset}")
        print("=" * 80)

        asset_training = training[
            training["asset"] == asset
        ].copy()

        asset_oos = oos[
            oos["asset"] == asset
        ].copy()

        asset_training_news = (
            training_with_news[
                training_with_news["asset"]
                == asset
            ].copy()
        )

        asset_oos_news = (
            oos_with_news[
                oos_with_news["asset"]
                == asset
            ].copy()
        )

        # ----------------------------------------------------
        # News-supported OOS sample
        # ----------------------------------------------------

        news_supported_oos = (
            asset_oos_news[
                asset_oos_news[
                    NEWS_FEATURES
                ]
                .notna()
                .all(axis=1)
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Matched training sample
        #
        # Training observations must also have complete
        # news features so that the Market + News model
        # can actually be fitted.
        # ----------------------------------------------------

        news_supported_training = (
            asset_training_news[
                asset_training_news[
                    NEWS_FEATURES
                ]
                .notna()
                .all(axis=1)
            ]
            .copy()
        )

        print()
        print("SAMPLE COUNTS")
        print("-" * 80)

        print(
            f"Total training rows: "
            f"{len(asset_training):,}"
        )

        print(
            f"Total OOS rows:      "
            f"{len(asset_oos):,}"
        )

        print(
            f"News-supported training rows: "
            f"{len(news_supported_training):,}"
        )

        print(
            f"News-supported OOS rows:      "
            f"{len(news_supported_oos):,}"
        )

        coverage = (
            len(news_supported_oos)
            / len(asset_oos)
            if len(asset_oos)
            else 0.0
        )

        print(
            f"OOS news coverage: "
            f"{coverage:.2%}"
        )

        if news_supported_training.empty:
            print()
            print(
                "NEWS MODEL STATUS"
            )
            print("-" * 80)
            print(
                "Skipped: no news-supported "
                "training observations."
            )
            continue

        if news_supported_oos.empty:
            print()
            print(
                "NEWS MODEL STATUS"
            )
            print("-" * 80)
            print(
                "Skipped: no news-supported "
                "OOS observations."
            )
            continue

        # ----------------------------------------------------
        # Target distributions
        # ----------------------------------------------------

        print_target_distribution(
            "MATCHED TRAINING TARGET DISTRIBUTION",
            news_supported_training,
        )

        print_target_distribution(
            "MATCHED OOS TARGET DISTRIBUTION",
            news_supported_oos,
        )

        # ----------------------------------------------------
        # MARKET-ONLY ON MATCHED OOS SAMPLE
        # ----------------------------------------------------

        market_training_matched = (
            asset_training[
                asset_training["timestamp"].isin(
                    news_supported_training[
                        "timestamp"
                    ]
                )
            ]
            .copy()
        )

        market_oos_matched = (
            asset_oos[
                asset_oos["timestamp"].isin(
                    news_supported_oos[
                        "timestamp"
                    ]
                )
            ]
            .copy()
        )

        market_result = fit_and_predict(
            market_training_matched,
            market_oos_matched,
            MARKET_FEATURES,
        )

        print_result(
            "MARKET-ONLY — MATCHED SAMPLE",
            market_result,
        )

        # ----------------------------------------------------
        # MARKET + NEWS
        # ----------------------------------------------------

        news_result = fit_and_predict(
            news_supported_training,
            news_supported_oos,
            MARKET_FEATURES
            + NEWS_FEATURES,
        )

        print_result(
            "MARKET + NEWS — MATCHED SAMPLE",
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

        print(f"Accuracy: {news_result['accuracy']:.4f} - {market_result['accuracy']:.4f}")

        print(
            f"Balanced accuracy: "
            f"{news_result['balanced_accuracy']:.4f} - {market_result['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro-F1: "
            f"{news_result['macro_f1']:.4f} - {market_result['macro_f1']:.4f}"
        )

        # ----------------------------------------------------
        # Confusion matrices
        # ----------------------------------------------------

        print()
        print(
            "MARKET-ONLY CONFUSION MATRIX"
        )
        print_confusion_matrix(
            market_result
        )

        print()
        print(
            "MARKET + NEWS CONFUSION MATRIX"
        )
        print_confusion_matrix(
            news_result
        )

    # --------------------------------------------------------
    # LOCKED HOLDOUT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOCKED HOLDOUT PROTECTION")
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

    print()
    print("=" * 80)
    print(
        "PHASE 2.3B MATCHED-SAMPLE "
        "EXPERIMENT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()