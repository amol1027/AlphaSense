from pathlib import Path

import pandas as pd

from src.features.trading_calendar import (
    NSEBSETradingCalendar,
    load_nse_bse_calendar,
)
from src.ingestion.market.audit import (
    MarketAuditResult,
    audit_market_data,
)


DATA_DIR = Path("data/raw/market")
REFERENCE_DIR = Path("data/reference")

ASSETS = {
    "RELIANCE": DATA_DIR / "phase1_reliance_15m.csv",
    "TCS": DATA_DIR / "phase1_tcs_15m.csv",
}

HOLIDAY_FILES = [
    REFERENCE_DIR / "nse_bse_holidays_2024.json",
    REFERENCE_DIR / "nse_bse_holidays_2025.json",
    REFERENCE_DIR / "nse_bse_holidays_2026.json",
]


def load_combined_calendar() -> NSEBSETradingCalendar:
    """
    Load all 2024-2026 NSE/BSE holidays and
    special sessions into one combined calendar.
    """

    holidays = set()
    special_sessions = {}

    for path in HOLIDAY_FILES:
        calendar = load_nse_bse_calendar(path)

        holidays.update(
            calendar.holidays
        )

        special_sessions.update(
            calendar.special_sessions
        )

    return NSEBSETradingCalendar(
        holidays=holidays,
        special_sessions=special_sessions,
    )

def print_asset_result(
    result: MarketAuditResult,
) -> None:
    print()
    print(result.asset)
    print("-" * 70)

    print(
        f"Rows:                       "
        f"{result.rows:,}"
    )

    print(
        f"Observed sessions:          "
        f"{result.observed_sessions:,}"
    )

    print(
        f"Expected sessions:          "
        f"{result.expected_sessions:,}"
    )

    print(
        f"Missing sessions:           "
        f"{len(result.missing_sessions):,}"
    )

    if result.missing_sessions:
        print(
            "  Missing dates:",
            ", ".join(
                str(value)
                for value in result.missing_sessions
            ),
        )

    print(
        f"Unexpected sessions:        "
        f"{len(result.unexpected_sessions):,}"
    )

    if result.unexpected_sessions:
        print(
            "  Unexpected dates:",
            ", ".join(
                str(value)
                for value in result.unexpected_sessions
            ),
        )

    print(
        f"Missing 15m bars:            "
        f"{result.missing_bars:,}"
    )

    print(
        f"Unexpected 15m bars:        "
        f"{result.unexpected_bars:,}"
    )

    print(
        f"Duplicate timestamps:       "
        f"{result.duplicate_timestamps:,}"
    )

    print(
        f"Misaligned timestamps:      "
        f"{result.misaligned_timestamps:,}"
    )

    print(
        f"Weekend bars:               "
        f"{result.weekend_bars:,}"
    )

    print(
        f"Holiday bars:               "
        f"{result.holiday_bars:,}"
    )

    print(
        f"Invalid numeric values:     "
        f"{result.invalid_numeric_values:,}"
    )

    print(
        f"OHLC violations:            "
        f"{result.ohlc_violations:,}"
    )

    print(
        f"STATUS:                     "
        f"{result.status}"
    )


def audit_cross_asset(
    dataframes: dict[str, pd.DataFrame],
) -> bool:
    """
    Compare timestamp coverage across assets.
    """

    if len(dataframes) < 2:
        return True

    print()
    print("CROSS-ASSET")
    print("-" * 70)

    timestamp_sets = {
        asset: set(
            pd.to_datetime(
                df["timestamp"],
                utc=True,
                errors="raise",
            )
        )
        for asset, df in dataframes.items()
    }

    assets = list(timestamp_sets)

    reference_asset = assets[0]
    reference_timestamps = timestamp_sets[
        reference_asset
    ]

    passed = True

    for asset in assets[1:]:
        current_timestamps = timestamp_sets[
            asset
        ]

        common = (
            reference_timestamps
            & current_timestamps
        )

        reference_only = (
            reference_timestamps
            - current_timestamps
        )

        current_only = (
            current_timestamps
            - reference_timestamps
        )

        print(
            f"{reference_asset} ↔ {asset}"
        )

        print(
            f"  Common timestamps:        "
            f"{len(common):,}"
        )

        print(
            f"  {reference_asset}-only:             "
            f"{len(reference_only):,}"
        )

        print(
            f"  {asset}-only:               "
            f"{len(current_only):,}"
        )

        if reference_only or current_only:
            passed = False

    return passed


def main() -> None:
    print("=" * 70)
    print("PHASE 1 MARKET DATA AUDIT")
    print("=" * 70)

    calendar = load_combined_calendar()

    print(
        f"Holiday dates loaded:       "
        f"{len(calendar.holidays)}"
    )

    dataframes = {}
    results = []

    for asset, path in ASSETS.items():
        print()
        print(
            f"Auditing {asset}: {path}"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Market data file not found: {path}"
            )

        df = pd.read_csv(path)

        dataframes[asset] = df

        result = audit_market_data(
            df=df,
            asset=asset,
            calendar=calendar,
        )

        results.append(result)

        print_asset_result(result)

    cross_asset_passed = audit_cross_asset(
        dataframes
    )

    print()
    print("=" * 70)

    asset_results_passed = all(
        result.passed
        for result in results
    )

    overall_passed = (
        asset_results_passed
        and cross_asset_passed
    )

    if overall_passed:
        print(
            "OVERALL STATUS: PASS"
        )
    else:
        print(
            "OVERALL STATUS: FAIL"
        )

    print("=" * 70)

    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()