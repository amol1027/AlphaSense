from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    positive_probability: float = Field(ge=0.0, le=1.0)
    neutral_probability: float = Field(ge=0.0, le=1.0)
    negative_probability: float = Field(ge=0.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)