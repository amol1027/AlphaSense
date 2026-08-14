from datetime import datetime

from pydantic import BaseModel, HttpUrl


class NewsArticle(BaseModel):
    asset: str
    published_at: datetime
    source: str
    headline: str
    text: str
    url: HttpUrl