from datetime import datetime

from pydantic import BaseModel, HttpUrl


class RedditPost(BaseModel):
    asset: str
    exchange: str
    published_at: datetime
    source: str
    headline: str
    text: str
    url: HttpUrl
    score: int
    comments: int