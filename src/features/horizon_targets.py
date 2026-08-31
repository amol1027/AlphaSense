import pandas as pd


def add_horizon_target(
    market_df: pd.DataFrame,
    horizon: pd.Timedelta,
) -> pd.DataFrame:
    """
    Add a future-return and direction target for an exact time horizon.

    The future close is taken from the same asset/exchange at exactly
    prediction timestamp + horizon.

    If no observation exists exactly at that future timestamp,
    the target is missing.
    """

    if horizon <= pd.Timedelta(0):
        raise ValueError("horizon must be positive")

    df = market_df.copy()

    group_columns = [
        "asset",
        "exchange",
    ]

    future = df[
        group_columns + [
            "timestamp",
            "close",
        ]
    ].copy()

    future = future.rename(
        columns={
            "timestamp": "future_timestamp",
            "close": "future_close",
        }
    )

    df["future_timestamp"] = (
        df["timestamp"] + horizon
    )

    df = df.merge(
        future,
        on=group_columns + ["future_timestamp"],
        how="left",
    )

    df["target_return"] = (
        (df["future_close"] - df["close"])
        / df["close"]
    )

    df["target_direction"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64",
    )

    valid_target = df["target_return"].notna()

    df.loc[
        valid_target,
        "target_direction",
    ] = (
        df.loc[
            valid_target,
            "target_return",
        ] > 0
    ).astype(int)

    return df