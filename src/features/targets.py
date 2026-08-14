import pandas as pd


def add_next_hour_target(
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the next-hour return and direction target.

    The next row's close is treated as the end of the
    future prediction window.
    """

    df = market_df.copy()

    group_columns = ["asset", "exchange"]

    df["future_close"] = (
        df.groupby(group_columns)["close"]
        .shift(-1)
    )

    df["target_return"] = (
        (df["future_close"] - df["close"])
        / df["close"]
    )

    df["target_direction"] = (
        df["target_return"] > 0
    ).astype(int)

    return df