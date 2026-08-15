from datetime import datetime

import pytest
from pydantic import ValidationError

from src.ingestion.market_schema import MarketBar


def test_valid_market_bar():
    bar = MarketBar(
        asset="TCS",
        exchange="NSE",
        timestamp=datetime(
            2026,
            8,
            15,
            10,
            15,
        ),
        open=3520.0,
        high=3535.0,
        low=3515.0,
        close=3530.0,
        volume=130000,
    )

    assert bar.asset == "TCS"
    assert bar.exchange == "NSE"
    assert bar.close == 3530.0
    assert bar.volume == 130000


def test_missing_required_field_is_rejected():
    with pytest.raises(ValidationError):
        MarketBar(
            asset="TCS",
            exchange="NSE",
            timestamp=datetime(
                2026,
                8,
                15,
                10,
                15,
            ),
            open=3520.0,
            high=3535.0,
            low=3515.0,
            close=3530.0,
        )