from pathlib import Path

import numpy as np
import pandas as pd

from src.features.horizon_targets import add_horizon_target


INPUT_PATH = Path(
    "data/raw/market/phase1_research_market_15m.csv"
)

FINAL_TEST_START = pd.Timestamp(
    "2026-08-10",
    tz="UTC",
)

HORIZONS = {
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
}

PERCENTILES = [
    25,
    50,
    60,
    70,
    75,
    80,
    90,
    95,
]


def load_market_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    duplicate_count = df.duplicated(
        ["asset", "exchange", "timestamp"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            f"market observations"
        )

    return df.sort_values(
        ["asset", "exchange", "timestamp"]
    ).reset_index(drop=True)


def describe_returns(
    returns: pd.Series,
) -> None:
    absolute_returns = returns.abs()

    print()
    print("RETURN DISTRIBUTION")
    print("-" * 80)

    print(
        f"Mean return   : {returns.mean():.6%}"
    )
    print(
        f"Median return : {returns.median():.6%}"
    )
    print(
        f"Std return    : {returns.std():.6%}"
    )

    print()
    print("ABSOLUTE RETURN PERCENTILES")
    print("-" * 80)

    percentile_values = np.percentile(
        absolute_returns,
        PERCENTILES,
    )

    for percentile, value in zip(
        PERCENTILES,
        percentile_values,
    ):
        print(
            f"P{percentile:<2}           : "
            f"{value:.6%}"
        )


def describe_threshold(
    returns: pd.Series,
    threshold: float,
) -> dict:
    up = returns > threshold
    down = returns < -threshold
    neutral = ~(up | down)

    total = len(returns)

    return {
        "threshold": threshold,
        "up": int(up.sum()),
        "down": int(down.sum()),
        "neutral": int(neutral.sum()),
        "up_pct": up.mean(),
        "down_pct": down.mean(),
        "neutral_pct": neutral.mean(),
        "directional_pct": (
            (up | down).mean()
        ),
    }


def main() -> None:
    print("=" * 80)
    print("PHASE 2.2 — THRESHOLDED TARGET DIAGNOSTIC")
    print("=" * 80)

    market = load_market_data()

    print()
    print(f"Input: {INPUT_PATH}")
    print(f"Market rows: {len(market):,}")
    print(
        f"Locked test excluded: "
        f">= {FINAL_TEST_START}"
    )

    for asset in sorted(
        market["asset"].dropna().unique()
    ):
        asset_market = market[
            market["asset"] == asset
        ].copy()

        print()
        print("=" * 80)
        print(f"ASSET: {asset}")
        print("=" * 80)

        for horizon_name, horizon in HORIZONS.items():

            target_df = add_horizon_target(
                asset_market,
                horizon,
            )

            # Construct targets with full historical
            # context, then exclude the locked test.
            target_df = target_df[
                target_df["timestamp"] < FINAL_TEST_START
            ].copy()

            returns = (
                target_df["target_return"]
                .dropna()
            )

            print()
            print("=" * 80)
            print(f"HORIZON: {horizon_name}")
            print("=" * 80)

            print(
                f"Valid development returns: "
                f"{len(returns):,}"
            )

            describe_returns(returns)

            print()
            print("THRESHOLD SCENARIOS")
            print("-" * 80)

            thresholds = np.percentile(
                returns.abs(),
                PERCENTILES,
            )

            rows = []

            for threshold in thresholds:
                rows.append(
                    describe_threshold(
                        returns,
                        threshold,
                    )
                )

            result = pd.DataFrame(rows)

            print(
                result.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.6f}",
                )
            )

    print()
    print("=" * 80)
    print("TEST PERIOD EXCLUSION CHECK")
    print("=" * 80)

    test_rows = market[
        market["timestamp"] >= FINAL_TEST_START
    ]

    print(
        f"Locked test rows excluded: "
        f"{len(test_rows):,}"
    )

    print(
        "No thresholds were selected using "
        "the locked holdout."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()