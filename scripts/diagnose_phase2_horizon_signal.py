from pathlib import Path

import pandas as pd

from src.features.horizon_targets import add_horizon_target
from src.features.market_features import (
    add_normalized_market_features,
)
from scripts.diagnose_phase1_feature_signal import safe_auc


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

FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
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


def main() -> None:
    print("=" * 80)
    print("PHASE 2.1B — MULTI-HORIZON FEATURE SIGNAL")
    print("=" * 80)

    market = load_market_data()

    market = add_normalized_market_features(
        market
    )

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

        rows = []

        for horizon_name, horizon in HORIZONS.items():

            # Generate the future target using the complete
            # historical context first.
            target_df = add_horizon_target(
                asset_market,
                horizon,
            )

            # Only after target construction do we exclude
            # the locked final holdout.
            target_df = target_df[
                target_df["timestamp"] < FINAL_TEST_START
            ].copy()

            for feature in FEATURES:

                if feature not in target_df.columns:
                    continue

                auc = safe_auc(
                    target_df["target_direction"],
                    target_df[feature],
                )

                valid = (
                    target_df["target_direction"].notna()
                    & target_df[feature].notna()
                )

                rows.append(
                    {
                        "horizon": horizon_name,
                        "feature": feature,
                        "samples": int(valid.sum()),
                        "auc": auc,
                        "auc_distance": (
                            abs(auc - 0.5)
                            if pd.notna(auc)
                            else float("nan")
                        ),
                    }
                )

        if not rows:
            raise RuntimeError(
                f"No signal diagnostic rows produced "
                f"for asset {asset}"
            )

        result = pd.DataFrame(rows)

        print()
        print("UNIVARIATE AUC")
        print("-" * 80)

        print(
            result[
                [
                    "horizon",
                    "feature",
                    "samples",
                    "auc",
                    "auc_distance",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        print()
        print("BEST FEATURE BY HORIZON")
        print("-" * 80)

        best = (
            result.sort_values(
                ["horizon", "auc_distance"],
                ascending=[True, False],
            )
            .groupby("horizon", as_index=False)
            .first()
        )

        print(
            best[
                [
                    "horizon",
                    "feature",
                    "auc",
                    "auc_distance",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        print()
        print("MEAN AUC DISTANCE BY HORIZON")
        print("-" * 80)

        horizon_summary = (
            result.groupby("horizon")["auc_distance"]
            .mean()
            .sort_values(ascending=False)
        )

        print(
            horizon_summary.to_string(
                float_format=lambda x: f"{x:.6f}"
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
        "Targets were constructed using complete "
        "historical context, then locked-test rows "
        "were excluded from the diagnostic."
    )

    print(
        "No model fitting or feature selection "
        "uses the locked holdout."
    )

    print("=" * 80)
if __name__ == "__main__":
    main()