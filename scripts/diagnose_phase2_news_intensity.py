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

NEWS_WINDOW = pd.Timedelta(minutes=60)

BURST_LOOKBACK = pd.Timedelta(days=1)

MIN_BASELINE_WINDOWS = 3


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

NEWS_INTENSITY_FEATURES = [
    "news_count",
    "news_burst",
    "sentiment_std",
    "positive_ratio",
    "negative_ratio",
    "sentiment_imbalance",
    "source_diversity",
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

    df = pd.read_csv(
        MARKET_PATH
    )

    required = {
        "asset",
        "exchange",
        "timestamp",
    }

    missing = required - set(
        df.columns
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
            f"Found {duplicate_count} "
            "duplicate market observations."
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
            "Missing required news columns: "
            f"{sorted(missing)}"
        )

    news["published_at"] = pd.to_datetime(
        news["published_at"],
        utc=True,
    )

    if "source" not in news.columns:
        news["source"] = "unknown"

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
# BUILD TARGET
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
# NEWS FEATURES FOR ONE PREDICTION
# ============================================================

def build_news_features_for_prediction(
    asset: str,
    timestamp: pd.Timestamp,
    news: pd.DataFrame,
) -> dict:

    current_start = (
        timestamp - NEWS_WINDOW
    )

    current_news = news[
        (news["asset"] == asset)
        & (
            news["published_at"]
            >= current_start
        )
        & (
            news["published_at"]
            <= timestamp
        )
    ]

    if current_news.empty:
        return {
            "sentiment_mean": np.nan,
            "sentiment_std": np.nan,
            "news_count": np.nan,
            "positive_ratio": np.nan,
            "negative_ratio": np.nan,
            "sentiment_imbalance": np.nan,
            "source_diversity": np.nan,
            "news_burst": np.nan,
        }

    sentiment = current_news[
        "sentiment_score"
    ]

    news_count = len(
        current_news
    )

    positive_ratio = (
        current_news[
            "positive_probability"
        ]
        >= 0.5
    ).mean()

    negative_ratio = (
        current_news[
            "negative_probability"
        ]
        >= 0.5
    ).mean()

    sentiment_imbalance = (
        positive_ratio
        - negative_ratio
    )

    source_diversity = (
        current_news[
            "source"
        ]
        .nunique()
    )

    # --------------------------------------------------------
    # Historical news baseline
    #
    # IMPORTANT:
    # Only observations strictly BEFORE the current
    # prediction window are used for the baseline.
    # --------------------------------------------------------

    baseline_end = (
        current_start
    )

    baseline_start = (
        baseline_end
        - BURST_LOOKBACK
    )

    historical_news = news[
        (news["asset"] == asset)
        & (
            news["published_at"]
            >= baseline_start
        )
        & (
            news["published_at"]
            < baseline_end
        )
    ]

    # Convert historical article timestamps into
    # one-hour bins.
    if historical_news.empty:

        baseline_count = np.nan

    else:

        historical_counts = (
            historical_news
            .set_index(
                "published_at"
            )
            .resample(
                "60min"
            )
            .size()
        )

        historical_counts = (
            historical_counts[
                historical_counts.index
                < current_start
            ]
        )

        if len(
            historical_counts
        ) < MIN_BASELINE_WINDOWS:

            baseline_count = np.nan

        else:

            baseline_count = (
                historical_counts
                .mean()
            )

    if (
        pd.isna(baseline_count)
        or baseline_count <= 0
    ):

        news_burst = np.nan

    else:

        news_burst = (
            news_count
            / baseline_count
        )

    return {
        "sentiment_mean": (
            sentiment.mean()
        ),
        "sentiment_std": (
            sentiment.std()
            if len(sentiment) > 1
            else 0.0
        ),
        "news_count": news_count,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
        "sentiment_imbalance": (
            sentiment_imbalance
        ),
        "source_diversity": (
            source_diversity
        ),
        "news_burst": news_burst,
    }


# ============================================================
# BUILD NEWS FEATURES
# ============================================================

def build_news_features(
    predictions: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, prediction in (
        predictions.iterrows()
    ):

        features = (
            build_news_features_for_prediction(
                prediction["asset"],
                prediction["timestamp"],
                news,
            )
        )

        rows.append(
            {
                "asset": prediction[
                    "asset"
                ],
                "exchange": prediction[
                    "exchange"
                ],
                "timestamp": prediction[
                    "timestamp"
                ],
                **features,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ATTACH NEWS FEATURES
# ============================================================

def attach_news_features(
    market: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:

    news_features = build_news_features(
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


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
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
# PRINT HELPERS
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
# INTENSITY DIAGNOSTICS
# ============================================================

def print_intensity_diagnostics(
    df: pd.DataFrame,
    asset: str,
) -> None:

    asset_df = df[
        (df["asset"] == asset)
        & df["target_class"].notna()
        & df["news_count"].notna()
    ].copy()

    print()
    print(
        "NEWS INTENSITY BY TARGET"
    )
    print("-" * 80)

    if asset_df.empty:
        print(
            "No usable news-supported observations."
        )
        return

    summary = (
        asset_df
        .groupby("target_class")[
            NEWS_INTENSITY_FEATURES
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
        .reindex(LABELS)
    )

    print(
        summary.to_string()
    )


def print_burst_distribution(
    df: pd.DataFrame,
    asset: str,
) -> None:

    asset_df = df[
        (df["asset"] == asset)
        & df["news_burst"].notna()
    ].copy()

    print()
    print(
        "NEWS BURST DISTRIBUTION"
    )
    print("-" * 80)

    if asset_df.empty:
        print(
            "No valid burst observations."
        )
        return

    print(
        asset_df[
            "news_burst"
        ]
        .describe()
        .to_string()
    )

    print()
    print(
        "BURST QUANTILES"
    )
    print("-" * 80)

    print(
        asset_df[
            "news_burst"
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
        .to_string()
    )


def print_density_target_relationship(
    df: pd.DataFrame,
    asset: str,
) -> None:

    asset_df = df[
        (df["asset"] == asset)
        & df["news_count"].notna()
        & df["target_class"].notna()
    ].copy()

    if asset_df.empty:
        return

    try:

        asset_df[
            "density_bin"
        ] = pd.qcut(
            asset_df["news_count"],
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
        asset_df[
            "density_bin"
        ],
        asset_df[
            "target_class"
        ],
        normalize="index",
    )

    table = table.reindex(
        columns=LABELS,
        fill_value=0.0,
    )

    print(
        table.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )


def print_burst_target_relationship(
    df: pd.DataFrame,
    asset: str,
) -> None:

    asset_df = df[
        (df["asset"] == asset)
        & df["news_burst"].notna()
        & df["target_class"].notna()
    ].copy()

    if asset_df.empty:
        return

    try:

        asset_df[
            "burst_bin"
        ] = pd.qcut(
            asset_df["news_burst"],
            q=4,
            duplicates="drop",
        )

    except ValueError:

        return

    print()
    print(
        "TARGET DISTRIBUTION BY NEWS BURST"
    )
    print("-" * 80)

    table = pd.crosstab(
        asset_df[
            "burst_bin"
        ],
        asset_df[
            "target_class"
        ],
        normalize="index",
    )

    table = table.reindex(
        columns=LABELS,
        fill_value=0.0,
    )

    print(
        table.to_string(
            float_format=lambda x:
            f"{x:.4f}"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "PHASE 2.6 — NEWS EVENT INTENSITY / REGIME SIGNAL"
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
        f"Horizon:      {HORIZON}"
    )

    print(
        f"Target threshold: "
        f"±{TARGET_THRESHOLD:.6%}"
    )

    print(
        f"News window:  {NEWS_WINDOW}"
    )

    print(
        f"Burst lookback: "
        f"{BURST_LOOKBACK}"
    )

    print(
        f"Locked start:  {LOCKED_START}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    data = build_target(
        market
    )

    # --------------------------------------------------------
    # DEVELOPMENT ONLY
    # --------------------------------------------------------

    development = data[
    (data["timestamp"] >= NEWS_TRAIN_START)
    & (data["timestamp"] < LOCKED_START)
    & (data["target_class"].notna())].copy()

    locked = data[
        data["timestamp"] >= LOCKED_START
    ].copy()

    print()
    print("=" * 80)
    print(
        "DEVELOPMENT DATA"
    )
    print("=" * 80)

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
        f"Locked rows: "
        f"{len(locked):,}"
    )

    # --------------------------------------------------------
    # BUILD NEWS FEATURES
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "BUILDING NEWS INTENSITY FEATURES"
    )
    print("=" * 80)

    enriched = attach_news_features(
        development,
        news,
    )

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    for asset in sorted(
        enriched["asset"].unique()
    ):

        print()
        print("=" * 80)
        print(
            f"ASSET: {asset}"
        )
        print("=" * 80)

        asset_df = enriched[
            enriched["asset"] == asset
        ].copy()

        total_rows = len(
            asset_df
        )

        news_supported = (
            asset_df[
                "news_count"
            ]
            .notna()
            .sum()
        )

        coverage = (
            news_supported / total_rows
            if total_rows
            else 0.0
        )

        print()
        print(
            "NEWS COVERAGE"
        )
        print("-" * 80)

        print(
            f"Supported rows: "
            f"{news_supported:,}/"
            f"{total_rows:,}"
        )

        print(
            f"Coverage: "
            f"{coverage:.2%}"
        )

        print_intensity_diagnostics(
            enriched,
            asset,
        )

        print_burst_distribution(
            enriched,
            asset,
        )

        print_density_target_relationship(
            enriched,
            asset,
        )

        print_burst_target_relationship(
            enriched,
            asset,
        )

        # ----------------------------------------------------
        # CHRONOLOGICAL MODEL SPLIT
        # ----------------------------------------------------

        asset_training = asset_df[
            asset_df["timestamp"]
            < OOS_START
        ].copy()

        asset_oos = asset_df[
            asset_df["timestamp"]
            >= OOS_START
        ].copy()

        print()
        print("=" * 80)
        print(
            "MODEL COMPARISON"
        )
        print("=" * 80)

        print()
        print(
            f"Training period: "
            f"{NEWS_TRAIN_START} → {OOS_START}"
        )

        print(
            f"OOS period: "
            f"{OOS_START} → {LOCKED_START}"
        )

        # ----------------------------------------------------
        # MARKET ONLY
        # ----------------------------------------------------

        market_result = evaluate_model(
            asset_training,
            asset_oos,
            MARKET_FEATURES,
        )

        print_result(
            "MARKET-ONLY",
            market_result,
        )

        # ----------------------------------------------------
        # MARKET + INTENSITY
        # ----------------------------------------------------

        intensity_result = evaluate_model(
            asset_training,
            asset_oos,
            MARKET_FEATURES
            + NEWS_INTENSITY_FEATURES,
        )

        print_result(
            "MARKET + NEWS INTENSITY",
            intensity_result,
        )

        # ----------------------------------------------------
        # CHANGE
        # ----------------------------------------------------

        print()
        print(
            "INCREMENTAL CHANGE"
        )
        print("-" * 80)

        print(
            f"Accuracy:            "
            f"{intensity_result['accuracy'] - market_result['accuracy']:+.4f}"
        )

        print(
            f"Balanced accuracy:   "
            f"{intensity_result['balanced_accuracy'] - market_result['balanced_accuracy']:+.4f}"
        )

        print(
            f"Macro-F1:            "
            f"{intensity_result['macro_f1'] - market_result['macro_f1']:+.4f}"
        )

    # --------------------------------------------------------
    # HOLDOUT PROTECTION
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
        "for feature construction, model fitting, "
        "or metric comparison."
    )

    print()
    print("=" * 80)
    print(
        "PHASE 2.6 NEWS EVENT INTENSITY "
        "EXPERIMENT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()