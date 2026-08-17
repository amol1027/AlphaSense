import pandas as pd


def add_next_hour_target(
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the one-hour-ahead return and direction target.

    The future close is taken from the same asset/exchange
    exactly one hour after the prediction timestamp.

    If no observation exists exactly one hour later,
    the target is missing.
    """

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
        df["timestamp"]
        + pd.Timedelta(hours=1)
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

    valid_target = (
        df["target_return"].notna()
    )

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