from functools import lru_cache

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .schemas import SentimentResult


FINBERT_MODEL_NAME = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def _load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(
        FINBERT_MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            FINBERT_MODEL_NAME
        )
    )

    model.eval()

    return tokenizer, model


def _result_from_probabilities(
    probabilities: torch.Tensor,
    labels: dict[str, int],
) -> SentimentResult:
    positive_probability = float(
        probabilities[
            labels["positive"]
        ].item()
    )

    neutral_probability = float(
        probabilities[
            labels["neutral"]
        ].item()
    )

    negative_probability = float(
        probabilities[
            labels["negative"]
        ].item()
    )

    sentiment_score = (
        positive_probability
        - negative_probability
    )

    return SentimentResult(
        positive_probability=positive_probability,
        neutral_probability=neutral_probability,
        negative_probability=negative_probability,
        sentiment_score=sentiment_score,
    )


class FinBERTSentimentProvider:
    """FinBERT-based financial sentiment provider."""

    def predict(
        self,
        text: str,
    ) -> SentimentResult:

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        results = self.predict_batch([text])

        return results[0]

    def predict_batch(
        self,
        texts: list[str],
    ) -> list[SentimentResult]:

        if not texts:
            return []

        if any(
            not text or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Texts cannot contain empty values."
            )

        tokenizer, model = _load_finbert()

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        labels = {
            str(label).lower(): index
            for index, label in (
                model.config.id2label.items()
            )
        }

        return [
            _result_from_probabilities(
                row,
                labels,
            )
            for row in probabilities
        ]