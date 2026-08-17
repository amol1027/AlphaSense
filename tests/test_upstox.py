from datetime import date

import pandas as pd
import pytest

from src.ingestion.market.upstox import (
    fetch_historical_candles,
)


class MockResponse:
    def __init__(
        self,
        payload,
        status_code=200,
    ):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(
                f"HTTP {self.status_code}"
            )

    def json(self):
        return self.payload


def test_historical_candles_are_chronologically_sorted(
    monkeypatch,
):
    payload = {
        "data": {
            "candles": [
                [
                    "2026-08-10T10:15:00+05:30",
                    101,
                    103,
                    100,
                    102,
                    2000,
                ],
                [
                    "2026-08-10T09:15:00+05:30",
                    100,
                    101,
                    99,
                    100.5,
                    1500,
                ],
            ]
        }
    }

    def mock_get(*args, **kwargs):
        return MockResponse(payload)

    monkeypatch.setattr(
        "src.ingestion.market.upstox.requests.get",
        mock_get,
    )

    result = fetch_historical_candles(
        "NSE_EQ|TEST",
        date(2026, 8, 10),
        date(2026, 8, 10),
    )

    assert len(result) == 2

    assert result[
        "timestamp"
    ].is_monotonic_increasing

    assert result.iloc[0]["open"] == 100
    assert result.iloc[1]["open"] == 101


def test_canonical_columns(monkeypatch):
    payload = {
        "data": {
            "candles": [
                [
                    "2026-08-10T09:15:00+05:30",
                    100,
                    101,
                    99,
                    100.5,
                    1500,
                ]
            ]
        }
    }

    def mock_get(*args, **kwargs):
        return MockResponse(payload)

    monkeypatch.setattr(
        "src.ingestion.market.upstox.requests.get",
        mock_get,
    )

    result = fetch_historical_candles(
        "NSE_EQ|TEST",
        date(2026, 8, 10),
        date(2026, 8, 10),
    )

    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    
    def mock_get(*args, **kwargs):
        return MockResponse(payload)

    import src.ingestion.market.upstox as upstox

    upstox.os.environ[
        "UPSTOX_ACCESS_TOKEN"
    ] = "test-token"

    monkeypatch = pytest.MonkeyPatch()

    monkeypatch.setattr(
        upstox.requests,
        "get",
        mock_get,
    )

    result = upstox.fetch_historical_candles(
        "NSE_EQ|TEST",
        date(2026, 8, 10),
        date(2026, 8, 10),
    )

    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    monkeypatch.undo()


def test_empty_response_returns_empty_dataframe(
    monkeypatch,
):
    payload = {
        "data": {
            "candles": []
        }
    }

    def mock_get(*args, **kwargs):
        return MockResponse(payload)

    monkeypatch.setattr(
        "src.ingestion.market.upstox.requests.get",
        mock_get,
    )

    result = fetch_historical_candles(
        "NSE_EQ|TEST",
        date(2026, 8, 10),
        date(2026, 8, 10),
    )

    assert result.empty

    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_invalid_interval_is_rejected():
    with pytest.raises(
        ValueError,
        match="interval_minutes",
    ):
        fetch_historical_candles(
            "NSE_EQ|TEST",
            date(2026, 8, 10),
            date(2026, 8, 10),
            interval_minutes=30,
        )


def test_invalid_date_range_is_rejected():
    with pytest.raises(
        ValueError,
        match="to_date",
    ):
        fetch_historical_candles(
            "NSE_EQ|TEST",
            date(2026, 8, 11),
            date(2026, 8, 10),
        )


def test_missing_access_token_is_rejected(
    monkeypatch,
):
    monkeypatch.delenv(
        "UPSTOX_ACCESS_TOKEN",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="UPSTOX_ACCESS_TOKEN",
    ):
        fetch_historical_candles(
            "NSE_EQ|TEST",
            date(2026, 8, 10),
            date(2026, 8, 10),
        )


def test_malformed_response_is_rejected(
    monkeypatch,
):
    payload = {
        "unexpected": {}
    }

    def mock_get(*args, **kwargs):
        return MockResponse(payload)

    monkeypatch.setattr(
        "src.ingestion.market.upstox.requests.get",
        mock_get,
    )

    with pytest.raises(
        ValueError,
        match="Unexpected Upstox",
    ):
        fetch_historical_candles(
            "NSE_EQ|TEST",
            date(2026, 8, 10),
            date(2026, 8, 10),
        )


def test_malformed_candle_is_rejected(
    monkeypatch,
):
    payload = {
        "data": {
            "candles": [
                [
                    "2026-08-10T09:15:00+05:30",
                    100,
                ]
            ]
        }
    }

    def mock_get(*args, **kwargs):
        return MockResponse(payload)

    monkeypatch.setattr(
        "src.ingestion.market.upstox.requests.get",
        mock_get,
    )

    with pytest.raises(
        ValueError,
        match="Malformed candle",
    ):
        fetch_historical_candles(
            "NSE_EQ|TEST",
            date(2026, 8, 10),
            date(2026, 8, 10),
        )