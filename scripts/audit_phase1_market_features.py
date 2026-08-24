from pathlib import Path

import pandas as pd


MARKET_PATH = Path(
    "data/raw/market/research_market_15m.csv"
)

FEATURE_PATH = Path(
    "data/processed/phase1_features.csv"
)

MARKET_FEATURES = [
    "return_15m",
    "return_30m",
    "return_1h",
    "high_low_range",
    "close_open_return",
    "volume_change",
]


def calculate_expected_market_features(
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Independently calculate market features.

    IMPORTANT:
    Calculations are performed on the complete market
    history before restricting to overlapping timestamps.
    This preserves the historical lag context required
    by return and volume features.
    """

    df = market_df.copy()

    df = df.sort_values(
        [
            "asset",
            "exchange",
            "timestamp",
        ]
    ).reset_index(drop=True)

    grouped_close = df.groupby(
        [
            "asset",
            "exchange",
        ],
        sort=False,
    )["close"]

    grouped_volume = df.groupby(
        [
            "asset",
            "exchange",
        ],
        sort=False,
    )["volume"]

    df["return_15m"] = (
        df["close"]
        / grouped_close.shift(1)
        - 1.0
    )

    df["return_30m"] = (
        df["close"]
        / grouped_close.shift(2)
        - 1.0
    )

    df["return_1h"] = (
        df["close"]
        / grouped_close.shift(4)
        - 1.0
    )

    df["high_low_range"] = (
        df["high"]
        - df["low"]
    ) / df["close"]

    df["close_open_return"] = (
        df["close"]
        / df["open"]
        - 1.0
    )

    previous_volume = grouped_volume.shift(1)

    df["volume_change"] = (
        df["volume"]
        / previous_volume
        - 1.0
    )

    return df


def main() -> None:
    print("=" * 70)
    print("PHASE 1 MARKET FEATURE CUTOFF AUDIT")
    print("=" * 70)

    if not MARKET_PATH.exists():
        raise FileNotFoundError(
            f"Market file not found: {MARKET_PATH}"
        )

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}"
        )

    market = pd.read_csv(
        MARKET_PATH
    )

    features = pd.read_csv(
        FEATURE_PATH
    )

    market["timestamp"] = pd.to_datetime(
        market["timestamp"],
        utc=True,
        errors="raise",
    )

    features["prediction_timestamp"] = (
        pd.to_datetime(
            features["prediction_timestamp"],
            utc=True,
            errors="raise",
        )
    )

    print()
    print(
        f"Market rows:   {len(market):,}"
    )

    print(
        f"Feature rows:  {len(features):,}"
    )

    failures = []

    # --------------------------------------------------------------
    # Independently calculate features on COMPLETE market dataset.
    # --------------------------------------------------------------

    expected = calculate_expected_market_features(
        market
    )

    for asset in sorted(
        features["asset"].dropna().unique()
    ):
        print()
        print("=" * 70)
        print(f"ASSET: {asset}")
        print("=" * 70)

        asset_features = features[
            features["asset"] == asset
        ].copy()

        asset_market = expected[
            expected["asset"] == asset
        ].copy()

        print(
            f"Feature rows:              "
            f"{len(asset_features):,}"
        )

        print(
            f"Market rows:               "
            f"{len(asset_market):,}"
        )

        feature_timestamps = set(
            asset_features[
                "prediction_timestamp"
            ]
        )

        market_timestamps = set(
            asset_market[
                "timestamp"
            ]
        )

        common_timestamps = (
            feature_timestamps
            & market_timestamps
        )

        feature_only_timestamps = (
            feature_timestamps
            - market_timestamps
        )

        print(
            f"Common timestamps:         "
            f"{len(common_timestamps):,}"
        )

        print(
            f"Feature-only timestamps:   "
            f"{len(feature_only_timestamps):,}"
        )

        if not common_timestamps:
            failures.append(
                f"{asset}: no overlapping "
                "timestamps."
            )
            continue

        # ----------------------------------------------------------
        # Restrict BOTH datasets only after calculations are done.
        # ----------------------------------------------------------

        actual = asset_features[
            asset_features[
                "prediction_timestamp"
            ].isin(common_timestamps)
        ].copy()

        expected_asset = asset_market[
            asset_market[
                "timestamp"
            ].isin(common_timestamps)
        ].copy()

        expected_asset = expected_asset.rename(
            columns={
                "timestamp":
                    "prediction_timestamp"
            }
        )

        merged = actual.merge(
            expected_asset[
                [
                    "prediction_timestamp",
                    *MARKET_FEATURES,
                ]
            ],
            on="prediction_timestamp",
            how="inner",
            suffixes=(
                "_actual",
                "_expected",
            ),
            validate="one_to_one",
        )

        # ----------------------------------------------------------
        # Compare independently calculated values.
        # ----------------------------------------------------------

        for feature in MARKET_FEATURES:

            actual_values = pd.to_numeric(
                merged[
                    f"{feature}_actual"
                ],
                errors="coerce",
            )

            expected_values = pd.to_numeric(
                merged[
                    f"{feature}_expected"
                ],
                errors="coerce",
            )

            both_nan = (
                actual_values.isna()
                & expected_values.isna()
            )

            comparable = (
                ~both_nan
                & actual_values.notna()
                & expected_values.notna()
            )

            if not comparable.any():
                print(
                    f"{feature:22s} "
                    "no comparable values"
                )
                continue

            differences = (
                actual_values[comparable]
                - expected_values[comparable]
            ).abs()

            max_difference = (
                differences.max()
            )

            print(
                f"{feature:22s} "
                f"max difference: "
                f"{max_difference:.12f}"
            )

            if max_difference > 1e-10:
                failures.append(
                    f"{asset}: {feature} differs "
                    "from independently calculated "
                    "historical-only values."
                )

        # ----------------------------------------------------------
        # Boundary spot check.
        # ----------------------------------------------------------

        print()
        print("BOUNDARY SPOT CHECKS")
        print("-" * 70)

        print(
            actual[
                [
                    "prediction_timestamp",
                    "return_15m",
                    "return_30m",
                    "return_1h",
                    "volume_change",
                ]
            ]
            .sort_values(
                "prediction_timestamp"
            )
            .head(8)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------------
    # Final result.
    # --------------------------------------------------------------

    print()
    print("=" * 70)

    if failures:
        print("AUDIT: FAIL")
        print("=" * 70)

        for failure in failures:
            print(
                f"- {failure}"
            )

        raise SystemExit(1)

    print("AUDIT: PASS")
    print("=" * 70)

    print()
    print(
        "All overlapping market features match "
        "independently calculated historical-only "
        "values."
    )

    print(
        "Lagged features were calculated using the "
        "complete available market history before "
        "timestamp comparison."
    )


if __name__ == "__main__":
    main()