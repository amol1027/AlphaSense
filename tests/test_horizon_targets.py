import pandas as pd

from src.features.horizon_targets import add_horizon_target


def make_market_df():
    timestamps = pd.date_range(
        "2026-08-10 09:15:00",
        periods=17,
        freq="15min",
    )

    return pd.DataFrame(
        {
            "asset": ["TCS"] * len(timestamps),
            "exchange": ["NSE"] * len(timestamps),
            "timestamp": timestamps,
            "close": [100 + i for i in range(len(timestamps))],
        }
    )


def test_30_minute_target_uses_exact_future_timestamp():
    df = make_market_df()

    result = add_horizon_target(df, pd.Timedelta(minutes=30))

    expected_return = (102 - 100) / 100

    assert result.loc[0, "target_return"] == expected_return
    assert result.loc[0, "target_direction"] == 1


def test_1_hour_target_matches_phase1_semantics():
    df = make_market_df()

    result = add_horizon_target(df, pd.Timedelta(hours=1))

    expected_return = (104 - 100) / 100

    assert result.loc[0, "target_return"] == expected_return
    assert result.loc[0, "target_direction"] == 1


def test_2_hour_target():
    df = make_market_df()

    result = add_horizon_target(df, pd.Timedelta(hours=2))

    expected_return = (108 - 100) / 100

    assert result.loc[0, "target_return"] == expected_return
    assert result.loc[0, "target_direction"] == 1


def test_4_hour_target():
    df = make_market_df()

    result = add_horizon_target(df, pd.Timedelta(hours=4))

    expected_return = (116 - 100) / 100

    assert result.loc[0, "target_return"] == expected_return
    assert result.loc[0, "target_direction"] == 1


def test_missing_future_timestamp_produces_missing_target():
    df = make_market_df()

    result = add_horizon_target(
        df,
        pd.Timedelta(hours=4),
    )

    assert pd.isna(result.loc[1, "target_return"])
    assert pd.isna(result.loc[1, "target_direction"])


def test_intermediate_bars_do_not_change_target():
    df = make_market_df()

    df.loc[1:3, "close"] = [1000, 2000, 3000]

    result = add_horizon_target(
        df,
        pd.Timedelta(hours=1),
    )

    expected_return = (104 - 100) / 100

    assert result.loc[0, "target_return"] == expected_return