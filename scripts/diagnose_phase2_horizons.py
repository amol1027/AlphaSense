from pathlib import Path

import pandas as pd

from src.features.horizon_targets import add_horizon_target
from src.ingestion import market


MARKET_PATH = Path(
    "data/raw/market/phase1_research_market_15m.csv"
)
HORIZONS = {
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
}

LOCKED_HOLDOUT_START = pd.Timestamp(
    "2026-08-10",
    tz="UTC",
)


def load_market_data() -> pd.DataFrame:
    df = pd.read_csv(MARKET_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    duplicate_count = df.duplicated(
        ["asset", "exchange", "timestamp"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate market observations"
        )

    required_assets = {"RELIANCE", "TCS"}
    actual_assets = set(df["asset"].unique())

    if actual_assets != required_assets:
        raise ValueError(
            f"Unexpected assets: {sorted(actual_assets)}"
        )

    df = df.sort_values(
        ["asset", "exchange", "timestamp"]
    ).reset_index(drop=True)

    return df

def describe_target(
    df: pd.DataFrame,
    horizon_name: str,
    horizon: pd.Timedelta,
) -> None:
    result = add_horizon_target(
        df,
        horizon,
    )

    development = result[
        result["timestamp"] < LOCKED_HOLDOUT_START
    ].copy()

    development = development[
        development["target_return"].notna()
    ].copy()

    print()
    print("=" * 80)
    print(f"HORIZON: {horizon_name}")
    print("=" * 80)

    for asset, asset_df in development.groupby("asset"):
        returns = asset_df["target_return"]
        direction = asset_df["target_direction"]

        up_count = int((direction == 1).sum())
        down_count = int((direction == 0).sum())
        total = up_count + down_count

        print()
        print(f"ASSET: {asset}")
        print("-" * 80)

        print(f"Valid observations : {total:,}")
        print(
            f"UP                 : {up_count:,} "
            f"({up_count / total:.2%})"
        )
        print(
            f"DOWN               : {down_count:,} "
            f"({down_count / total:.2%})"
        )

        print()
        print("Future return statistics:")
        print(f"Mean               : {returns.mean():.6%}")
        print(f"Median             : {returns.median():.6%}")
        print(f"Std                 : {returns.std():.6%}")
        print(f"Min                 : {returns.min():.6%}")
        print(f"Max                 : {returns.max():.6%}")


def main() -> None:
    market = load_market_data()

    print("=" * 80)
    print("PHASE 2.1A — TARGET HORIZON DIAGNOSTIC")
    print("=" * 80)
    print()
    print(f"Market source: {MARKET_PATH}")
    print(f"Market rows: {len(market):,}")
    print()
    print("Rows by asset:")
    print(market.groupby("asset").size().to_string())
    print(
        "Development period: "
        f"{market['timestamp'].min()} → "
        f"{LOCKED_HOLDOUT_START}"
    )
    print(
        "Locked holdout excluded from statistics: "
        f">= {LOCKED_HOLDOUT_START}"
    )

    for horizon_name, horizon in HORIZONS.items():
        describe_target(
            market,
            horizon_name,
            horizon,
        )


if __name__ == "__main__":
    main()