from pathlib import Path

import pandas as pd


MARKET_DIR = Path("data/raw/market")

INPUTS = [
    MARKET_DIR / "phase1_tcs_15m.csv",
    MARKET_DIR / "phase1_reliance_15m.csv",
]

OUTPUT = MARKET_DIR / "phase1_research_market_15m.csv"


def main():
    frames = []

    for path in INPUTS:
        df = pd.read_csv(
            path,
            parse_dates=["timestamp"],
        )

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "asset",
            "exchange",
            "timestamp",
        ]
    ).reset_index(drop=True)

    combined.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"Rows: {len(combined)}")
    print(
        "Assets:",
        combined["asset"]
        .value_counts()
        .to_dict(),
    )
    print(
        "Duplicates:",
        combined.duplicated(
            subset=[
                "asset",
                "exchange",
                "timestamp",
            ]
        ).sum(),
    )
    print(
        "Output:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()