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

NEWS_WINDOWS = [
    pd.Timedelta(minutes=30),
    pd.Timedelta(hours=1),
    pd.Timedelta(hours=2),
    pd.Timedelta(hours=4),
]


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
    window: pd.Timedelta,
) -> pd.DataFrame:

    rows = []

    for _, prediction in predictions.iterrows():

        asset = prediction["asset"]
        timestamp = prediction["timestamp"]

        window_start = (
            timestamp - window
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
# ATTACH NEWS
# ============================================================

def attach_news_features(
    market_data: pd.DataFrame,
    news: pd.DataFrame,
    window: pd.Timedelta,
) -> pd.DataFrame:

    news_features = (
        aggregate_news_for_predictions(
            market_data,
            news,
            window,
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
) -> dict:

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
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "PHASE 2.4 — NEWS WINDOW ROBUSTNESS"
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
        f"Training period: "
        f"{NEWS_TRAIN_START} → {OOS_START}"
    )

    print(
        f"OOS period: "
        f"{OOS_START} → {LOCKED_START}"
    )

    print(
        f"Locked period: >= {LOCKED_START}"
    )

    # --------------------------------------------------------
    # BUILD TARGET
    # --------------------------------------------------------

    data = build_target(
        market
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
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
        data["timestamp"] >= LOCKED_START
    ].copy()

    print()
    print("=" * 80)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 80)

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
    # STORE RESULTS
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # ASSET LOOP
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
        # Evaluate each news window
        # ----------------------------------------------------

        for window in NEWS_WINDOWS:

            window_label = (
                f"{int(window.total_seconds() / 60)}m"
            )

            print()
            print("=" * 80)
            print(
                f"NEWS WINDOW: {window_label}"
            )
            print("=" * 80)

            # -----------------------------------------------
            # Build news features
            # -----------------------------------------------

            training_news = (
                attach_news_features(
                    asset_training,
                    news,
                    window,
                )
            )

            oos_news = (
                attach_news_features(
                    asset_oos,
                    news,
                    window,
                )
            )

            # -----------------------------------------------
            # News-supported observations
            # -----------------------------------------------

            supported_training = (
                training_news[
                    training_news[
                        NEWS_FEATURES
                    ]
                    .notna()
                    .all(axis=1)
                ]
                .copy()
            )

            supported_oos = (
                oos_news[
                    oos_news[
                        NEWS_FEATURES
                    ]
                    .notna()
                    .all(axis=1)
                ]
                .copy()
            )

            print()
            print(
                "NEWS COVERAGE"
            )
            print("-" * 80)

            training_coverage = (
                len(supported_training)
                / len(asset_training)
                if len(asset_training)
                else 0.0
            )

            oos_coverage = (
                len(supported_oos)
                / len(asset_oos)
                if len(asset_oos)
                else 0.0
            )

            print(
                f"Training: "
                f"{len(supported_training):,}/"
                f"{len(asset_training):,} "
                f"({training_coverage:.2%})"
            )

            print(
                f"OOS:      "
                f"{len(supported_oos):,}/"
                f"{len(asset_oos):,} "
                f"({oos_coverage:.2%})"
            )

            # -----------------------------------------------
            # Require usable samples
            # -----------------------------------------------

            if (
                supported_training.empty
                or supported_oos.empty
            ):

                print()
                print(
                    "STATUS: SKIPPED"
                )
                print(
                    "Insufficient news-supported "
                    "observations."
                )

                continue

            # -----------------------------------------------
            # MATCHED MARKET SAMPLE
            # -----------------------------------------------

            matched_training = (
                asset_training[
                    asset_training["timestamp"].isin(
                        supported_training[
                            "timestamp"
                        ]
                    )
                ]
                .copy()
            )

            matched_oos = (
                asset_oos[
                    asset_oos["timestamp"].isin(
                        supported_oos[
                            "timestamp"
                        ]
                    )
                ]
                .copy()
            )

            # -----------------------------------------------
            # MARKET-ONLY
            # -----------------------------------------------

            market_result = fit_and_predict(
                matched_training,
                matched_oos,
                MARKET_FEATURES,
            )

            # -----------------------------------------------
            # MARKET + NEWS
            # -----------------------------------------------

            news_result = fit_and_predict(
                supported_training,
                supported_oos,
                MARKET_FEATURES
                + NEWS_FEATURES,
            )

            # -----------------------------------------------
            # INCREMENTAL CHANGE
            # -----------------------------------------------

            accuracy_change = (
                news_result["accuracy"]
                - market_result["accuracy"]
            )

            balanced_change = (
                news_result["balanced_accuracy"]
                - market_result["balanced_accuracy"]
            )

            macro_f1_change = (
                news_result["macro_f1"]
                - market_result["macro_f1"]
            )

            print()
            print(
                "MARKET-ONLY — MATCHED SAMPLE"
            )
            print("-" * 80)

            print(
                f"Training rows:       "
                f"{market_result['training_rows']:,}"
            )

            print(
                f"OOS rows:            "
                f"{market_result['oos_rows']:,}"
            )

            print(
                f"Accuracy:            "
                f"{market_result['accuracy']:.4f}"
            )

            print(
                f"Balanced accuracy:   "
                f"{market_result['balanced_accuracy']:.4f}"
            )

            print(
                f"Macro-F1:            "
                f"{market_result['macro_f1']:.4f}"
            )

            print()
            print(
                "MARKET + NEWS — MATCHED SAMPLE"
            )
            print("-" * 80)

            print(
                f"Training rows:       "
                f"{news_result['training_rows']:,}"
            )

            print(
                f"OOS rows:            "
                f"{news_result['oos_rows']:,}"
            )

            print(
                f"Accuracy:            "
                f"{news_result['accuracy']:.4f}"
            )

            print(
                f"Balanced accuracy:   "
                f"{news_result['balanced_accuracy']:.4f}"
            )

            print(
                f"Macro-F1:            "
                f"{news_result['macro_f1']:.4f}"
            )

            print()
            print(
                "INCREMENTAL CHANGE"
            )
            print("-" * 80)

            print(
                f"Accuracy:            "
                f"{accuracy_change:+.4f}"
            )

            print(
                f"Balanced accuracy:   "
                f"{balanced_change:+.4f}"
            )

            print(
                f"Macro-F1:            "
                f"{macro_f1_change:+.4f}"
            )

            results.append(
                {
                    "asset": asset,
                    "window": window_label,
                    "training_rows": (
                        news_result[
                            "training_rows"
                        ]
                    ),
                    "oos_rows": (
                        news_result[
                            "oos_rows"
                        ]
                    ),
                    "training_coverage": (
                        training_coverage
                    ),
                    "oos_coverage": (
                        oos_coverage
                    ),
                    "market_accuracy": (
                        market_result[
                            "accuracy"
                        ]
                    ),
                    "news_accuracy": (
                        news_result[
                            "accuracy"
                        ]
                    ),
                    "accuracy_change": (
                        accuracy_change
                    ),
                    "market_balanced_accuracy": (
                        market_result[
                            "balanced_accuracy"
                        ]
                    ),
                    "news_balanced_accuracy": (
                        news_result[
                            "balanced_accuracy"
                        ]
                    ),
                    "balanced_accuracy_change": (
                        balanced_change
                    ),
                    "market_macro_f1": (
                        market_result[
                            "macro_f1"
                        ]
                    ),
                    "news_macro_f1": (
                        news_result[
                            "macro_f1"
                        ]
                    ),
                    "macro_f1_change": (
                        macro_f1_change
                    ),
                }
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "PHASE 2.4 SUMMARY"
    )
    print("=" * 80)

    if results:

        summary = pd.DataFrame(
            results
        )

        print()

        print(
            summary[
                [
                    "asset",
                    "window",
                    "training_rows",
                    "oos_rows",
                    "oos_coverage",
                    "accuracy_change",
                    "balanced_accuracy_change",
                    "macro_f1_change",
                ]
            ]
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No valid window evaluations."
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
        "for news-window selection, model "
        "fitting, or metric comparison."
    )

    print()
    print("=" * 80)
    print(
        "PHASE 2.4 NEWS WINDOW "
        "ROBUSTNESS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()