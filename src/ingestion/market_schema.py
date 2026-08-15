from datetime import datetime

from pydantic import BaseModel


class MarketBar(BaseModel):
    asset: str
    exchange: str
    timestamp: datetime

    open: float
    high: float
    low: float
    close: float
    volume: int