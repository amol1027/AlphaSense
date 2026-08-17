from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.features.targets import add_next_hour_target


MARKET_DIR = Path("data/raw/market")


def inspect_asset(symbol: str):
    path = (
        MARKET_DIR
        / f"{symbol.lower()}_15m.csv"
    )

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )

    df = df.sort_values(
        ["asset", "exchange", "timestamp"]
    ).reset_index(drop=True)

    result = add_next_hour_target(df)

    print(f"\n=== {symbol} ===")
    print("Rows:", len(result))
    print(
        "Date range:",
        result["timestamp"].min(),
        "→",
        result["timestamp"].max(),
    )

    print(
        "Target rows:",
        result["target_direction"].notna().sum(),
    )

    print(
        "Missing targets:",
        result["target_direction"].isna().sum(),
    )

    print(
        "Class distribution:"
    )
    print(
        result["target_direction"]
        .value_counts(dropna=True)
        .sort_index()
        .to_string()
    )

    print("\nSample:")
    print(
        result[
            [
                "timestamp",
                "close",
                "future_close",
                "target_return",
                "target_direction",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


def main():
    inspect_asset("TCS")
    inspect_asset("RELIANCE")


if __name__ == "__main__":
    main()