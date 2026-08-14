from .schemas import SentimentResult


POSITIVE_WORDS = {
    "strong",
    "growth",
    "beats",
    "above",
    "robust",
    "expansion",
}

NEGATIVE_WORDS = {
    "concerns",
    "constraints",
    "pressure",
    "regulatory",
    "scrutiny",
}


class DummySentimentProvider:
    """Temporary sentiment provider used until FinBERT is added."""

    def predict(self, text: str) -> SentimentResult:
        text_lower = text.lower()

        positive_count = sum(
            word in text_lower for word in POSITIVE_WORDS
        )

        negative_count = sum(
            word in text_lower for word in NEGATIVE_WORDS
        )

        if positive_count > negative_count:
            positive_probability = 0.80
            neutral_probability = 0.15
            negative_probability = 0.05

        elif negative_count > positive_count:
            positive_probability = 0.05
            neutral_probability = 0.15
            negative_probability = 0.80

        else:
            positive_probability = 0.10
            neutral_probability = 0.80
            negative_probability = 0.10

        sentiment_score = (
            positive_probability - negative_probability
        )

        return SentimentResult(
            positive_probability=positive_probability,
            neutral_probability=neutral_probability,
            negative_probability=negative_probability,
            sentiment_score=sentiment_score,
        )