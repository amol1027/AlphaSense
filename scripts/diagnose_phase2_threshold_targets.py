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

HORIZON = pd.Timedelta(hours=1)

THRESHOLD_RULES = [
    "pooled_p50",
    "asset_p50",
    "pooled_p75",
    "asset_p75",
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


def classify_return(
    returns: pd.Series,
    threshold: float,
) -> pd.Series:
    target = pd.Series(
        "NEUTRAL",
        index=returns.index,
        dtype="string",
    )

    target.loc[
        returns > threshold
    ] = "UP"

    target.loc[
        returns < -threshold
    ] = "DOWN"

    return target


def describe_target(
    target: pd.Series,
) -> dict:
    counts = target.value_counts()

    total = len(target)

    up = int(counts.get("UP", 0))
    down = int(counts.get("DOWN", 0))
    neutral = int(counts.get("NEUTRAL", 0))

    return {
        "samples": total,
        "up": up,
        "down": down,
        "neutral": neutral,
        "up_pct": up / total,
        "down_pct": down / total,
        "neutral_pct": neutral / total,
        "directional_pct": (
            (up + down) / total
        ),
    }


def main() -> None:
    print("=" * 80)
    print("PHASE 2.2A — THRESHOLD TARGET USABILITY")
    print("=" * 80)

    market = load_market_data()

    # Construct the 1-hour target using the complete
    # market history first.
    target_df = add_horizon_target(
        market,
        HORIZON,
    )

    # Only development-period prediction timestamps
    # are used for threshold diagnosis.
    development = target_df[
        target_df["timestamp"] < FINAL_TEST_START
    ].copy()

    development = development[
        development["target_return"].notna()
    ].copy()

    print()
    print(f"Input: {INPUT_PATH}")
    print(f"Market rows: {len(market):,}")
    print(
        f"Development target rows: "
        f"{len(development):,}"
    )
    print(
        f"Locked test excluded: "
        f">= {FINAL_TEST_START}"
    )

    # ------------------------------------------------------------------
    # Threshold calculation
    # ------------------------------------------------------------------

    pooled_absolute_returns = (
        development["target_return"].abs()
    )

    pooled_p50 = float(
        pooled_absolute_returns.quantile(0.50)
    )

    pooled_p75 = float(
        pooled_absolute_returns.quantile(0.75)
    )

    asset_thresholds = {}

    for asset, asset_df in development.groupby(
        "asset"
    ):
        absolute_returns = (
            asset_df["target_return"].abs()
        )

        asset_thresholds[asset] = {
            "p50": float(
                absolute_returns.quantile(0.50)
            ),
            "p75": float(
                absolute_returns.quantile(0.75)
            ),
        }

    print()
    print("=" * 80)
    print("THRESHOLD DEFINITIONS")
    print("=" * 80)

    print()
    print(
        f"Pooled P50: "
        f"{pooled_p50:.6%}"
    )

    print(
        f"Pooled P75: "
        f"{pooled_p75:.6%}"
    )

    print()
    print("Asset-specific thresholds:")

    for asset in sorted(
        asset_thresholds
    ):
        values = asset_thresholds[asset]

        print(
            f"{asset}: "
            f"P50={values['p50']:.6%}, "
            f"P75={values['p75']:.6%}"
        )

    # ------------------------------------------------------------------
    # Evaluate threshold rules
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("THRESHOLD USABILITY")
    print("=" * 80)

    for asset in sorted(
        development["asset"].unique()
    ):
        asset_df = development[
            development["asset"] == asset
        ].copy()

        print()
        print("=" * 80)
        print(f"ASSET: {asset}")
        print("=" * 80)

        threshold_map = {
            "pooled_p50": pooled_p50,
            "asset_p50": asset_thresholds[asset]["p50"],
            "pooled_p75": pooled_p75,
            "asset_p75": asset_thresholds[asset]["p75"],
        }

        rows = []

        for rule in THRESHOLD_RULES:

            threshold = threshold_map[rule]

            target = classify_return(
                asset_df["target_return"],
                threshold,
            )

            description = describe_target(
                target
            )

            rows.append(
                {
                    "rule": rule,
                    "threshold": threshold,
                    **description,
                }
            )

        result = pd.DataFrame(rows)

        print(
            result.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6f}"
                ),
            )
        )

    # ------------------------------------------------------------------
    # Cross-asset consistency
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CROSS-ASSET CLASS-BALANCE CHECK")
    print("=" * 80)

    rows = []

    for rule in THRESHOLD_RULES:

        if rule == "pooled_p50":
            thresholds = {
                asset: pooled_p50
                for asset in asset_thresholds
            }

        elif rule == "pooled_p75":
            thresholds = {
                asset: pooled_p75
                for asset in asset_thresholds
            }

        elif rule == "asset_p50":
            thresholds = {
                asset: values["p50"]
                for asset, values
                in asset_thresholds.items()
            }

        else:
            thresholds = {
                asset: values["p75"]
                for asset, values
                in asset_thresholds.items()
            }

        for asset in sorted(
            thresholds
        ):
            asset_df = development[
                development["asset"] == asset
            ]

            target = classify_return(
                asset_df["target_return"],
                thresholds[asset],
            )

            description = describe_target(
                target
            )

            rows.append(
                {
                    "rule": rule,
                    "asset": asset,
                    "threshold": thresholds[asset],
                    "up_pct": description["up_pct"],
                    "down_pct": description["down_pct"],
                    "neutral_pct": description["neutral_pct"],
                    "directional_pct": description[
                        "directional_pct"
                    ],
                }
            )

    consistency = pd.DataFrame(rows)

    print(
        consistency.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ------------------------------------------------------------------
    # Explicit integrity checks
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("INTEGRITY CHECKS")
    print("=" * 80)

    if len(development) == 0:
        raise RuntimeError(
            "No development target rows available."
        )

    if development["target_return"].isna().any():
        raise RuntimeError(
            "Unexpected missing target returns."
        )

    test_rows = target_df[
        target_df["timestamp"] >= FINAL_TEST_START
    ]

    print(
        f"Locked test rows present in source: "
        f"{len(test_rows):,}"
    )

    print(
        "Locked test rows were excluded from "
        "all threshold calculations."
    )

    print()
    print("PHASE 2.2A DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()