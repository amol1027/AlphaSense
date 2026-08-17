from datetime import date

import pandas as pd
import pytest

from src.ingestion.market.downloader import (
    download_market_data,
)


def make_market_data():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-08-10 09:15",
                    "2026-08-10 09:30",
                ]
            ),
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [1000, 1200],
        }
    )


def test_download_validates_and_saves(
    monkeypatch,
    tmp_path,
):
    expected = make_market_data()

    def mock_fetch(**kwargs):
        return expected

    monkeypatch.setattr(
        "src.ingestion.market.downloader.fetch_historical_candles",
        mock_fetch,
    )

    output_path = (
        tmp_path / "market.csv"
    )

    result = download_market_data(
        instrument_key="NSE_EQ|TEST",
        asset="TCS",
        exchange="NSE",
        from_date=date(2026, 8, 10),
        to_date=date(2026, 8, 10),
        output_path=output_path,
    )

    assert len(result) == 2
    assert output_path.exists()

    saved = pd.read_csv(
        output_path
    )

    assert len(saved) == 2
    assert list(saved.columns) == [
    "asset",
    "exchange",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",]


def test_empty_provider_response_is_rejected(
    monkeypatch,
    tmp_path,
):
    def mock_fetch(**kwargs):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    monkeypatch.setattr(
        "src.ingestion.market.downloader.fetch_historical_candles",
        mock_fetch,
    )

    with pytest.raises(
        ValueError,
        match="No market data",
    ):
        download_market_data(
    instrument_key="NSE_EQ|TEST",
    asset="TCS",
    exchange="NSE",
    from_date=date(2026, 8, 10),
    to_date=date(2026, 8, 10),
    output_path=tmp_path / "market.csv",
)


def test_invalid_date_range_is_rejected(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="to_date",
    ):
        download_market_data(
            instrument_key="NSE_EQ|TEST",
            asset="TCS",
            exchange="NSE",
            from_date=date(2026, 8, 11),
            to_date=date(2026, 8, 10),
            output_path=tmp_path / "market.csv",
        )

def test_asset_and_exchange_are_added(
    monkeypatch,
    tmp_path,
):
    expected = make_market_data()

    def mock_fetch(**kwargs):
        return expected

    monkeypatch.setattr(
        "src.ingestion.market.downloader.fetch_historical_candles",
        mock_fetch,
    )

    output_path = (
        tmp_path / "market.csv"
    )

    result = download_market_data(
        instrument_key="NSE_EQ|TEST",
        asset="TCS",
        exchange="NSE",
        from_date=date(2026, 8, 10),
        to_date=date(2026, 8, 10),
        output_path=output_path,
    )

    assert set(result["asset"]) == {"TCS"}
    assert set(result["exchange"]) == {"NSE"}

    assert list(result.columns) == [
        "asset",
        "exchange",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_empty_asset_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="asset cannot be empty",
    ):
        download_market_data(
            instrument_key="NSE_EQ|TEST",
            asset="",
            exchange="NSE",
            from_date=date(2026, 8, 10),
            to_date=date(2026, 8, 10),
            output_path=tmp_path / "market.csv",
        )


def test_empty_exchange_is_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="exchange cannot be empty",
    ):
        download_market_data(
            instrument_key="NSE_EQ|TEST",
            asset="TCS",
            exchange="",
            from_date=date(2026, 8, 10),
            to_date=date(2026, 8, 10),
            output_path=tmp_path / "market.csv",
        )