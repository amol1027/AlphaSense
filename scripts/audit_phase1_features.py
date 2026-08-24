from pathlib import Path

import pandas as pd

from src.ingestion.feature_audit import (
    FeatureAuditResult,
    audit_feature_data,
)


INPUT_PATH = Path(
    "data/processed/phase1_features.csv"
)


def print_result(
    result: FeatureAuditResult,
) -> None:
    print()
    print("DATA QUALITY")
    print("-" * 70)

    print(
        f"Rows:                       "
        f"{result.rows:,}"
    )

    print(
        f"Date range:                 "
        f"{result.start_timestamp}"
        f" -> "
        f"{result.end_timestamp}"
    )

    print(
        f"Invalid timestamps:         "
        f"{result.invalid_timestamps:,}"
    )

    print(
        f"Null timestamps:            "
        f"{result.null_timestamps:,}"
    )

    print(
        f"Duplicate feature keys:     "
        f"{result.duplicate_keys:,}"
    )

    print(
        f"Invalid assets:             "
        f"{result.invalid_assets:,}"
    )

    print(
        f"Invalid exchanges:          "
        f"{result.invalid_exchanges:,}"
    )

    print(
        f"Invalid numeric values:     "
        f"{result.invalid_numeric_values:,}"
    )

    print(
        f"Infinite numeric values:    "
        f"{result.infinite_numeric_values:,}"
    )

    print(
        f"Missing targets:            "
        f"{result.missing_targets:,}"
    )

    print(
        f"Invalid targets:            "
        f"{result.invalid_targets:,}"
    )

    print(
        f"Invalid sentiment features:"
        f" {result.invalid_sentiment_features:,}"
    )

    print(
        f"Invalid Reddit features:    "
        f"{result.invalid_reddit_features:,}"
    )

    print(
        f"Weekend predictions:        "
        f"{result.weekend_predictions:,}"
    )

    print(
        f"Invalid target alignment:   "
        f"{result.invalid_target_alignment:,}"
    )

    print(
        f"STATUS:                     "
        f"{result.status}"
    )


def main() -> None:
    print("=" * 70)
    print("PHASE 1 FEATURE DATA AUDIT")
    print("=" * 70)

    print(
        f"Input: {INPUT_PATH}"
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    result = audit_feature_data(df)

    print_result(result)

    print()
    print("ASSETS")
    print("-" * 70)

    print(
        df["asset"]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 70)

    if result.passed:
        print(
            "OVERALL STATUS: PASS"
        )
    else:
        print(
            "OVERALL STATUS: FAIL"
        )

    print("=" * 70)

    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()