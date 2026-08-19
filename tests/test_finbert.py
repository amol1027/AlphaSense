import torch

from src.sentiment.finbert import (
    FinBERTSentimentProvider,
)


class FakeConfig:
    id2label = {
        0: "positive",
        1: "negative",
        2: "neutral",
    }


class FakeModel:
    config = FakeConfig()

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = kwargs[
            "input_ids"
        ].shape[0]

        class Output:
            logits = torch.tensor(
                [
                    [2.0, 1.0, 0.0]
                    for _ in range(batch_size)
                ]
            )

        return Output()


class FakeTokenizer:
    def __call__(
        self,
        text,
        return_tensors,
        truncation,
        max_length,
        padding=False,
    ):
        batch_size = (
            len(text)
            if isinstance(text, list)
            else 1
        )

        return {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3]
                    for _ in range(batch_size)
                ]
            )
        }


def test_finbert_returns_probabilities(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.sentiment.finbert._load_finbert",
        lambda: (
            FakeTokenizer(),
            FakeModel(),
        ),
    )

    provider = FinBERTSentimentProvider()

    result = provider.predict(
        "TCS reports strong growth."
    )

    assert (
        0.0
        <= result.positive_probability
        <= 1.0
    )

    assert (
        0.0
        <= result.neutral_probability
        <= 1.0
    )

    assert (
        0.0
        <= result.negative_probability
        <= 1.0
    )

    assert (
        abs(
            (
                result.positive_probability
                + result.neutral_probability
                + result.negative_probability
            )
            - 1.0
        )
        < 1e-6
    )


def test_finbert_score_is_positive_minus_negative(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.sentiment.finbert._load_finbert",
        lambda: (
            FakeTokenizer(),
            FakeModel(),
        ),
    )

    provider = FinBERTSentimentProvider()

    result = provider.predict(
        "Strong earnings growth."
    )

    assert result.sentiment_score == (
        result.positive_probability
        - result.negative_probability
    )


def test_finbert_rejects_empty_text():
    provider = FinBERTSentimentProvider()

    try:
        provider.predict("")
        assert False
    except ValueError as exc:
        assert "empty" in str(exc).lower()

def test_finbert_batch_returns_one_result_per_text(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.sentiment.finbert._load_finbert",
        lambda: (
            FakeTokenizer(),
            FakeModel(),
        ),
    )

    provider = FinBERTSentimentProvider()

    results = provider.predict_batch(
        [
            "Strong revenue growth.",
            "Weak earnings outlook.",
        ]
    )

    assert len(results) == 2

    for result in results:
        assert (
            0.0
            <= result.positive_probability
            <= 1.0
        )

        assert (
            0.0
            <= result.neutral_probability
            <= 1.0
        )

        assert (
            0.0
            <= result.negative_probability
            <= 1.0
        )


def test_finbert_batch_empty_input():
    provider = FinBERTSentimentProvider()

    assert provider.predict_batch([]) == []


def test_finbert_batch_rejects_empty_text(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.sentiment.finbert._load_finbert",
        lambda: (
            FakeTokenizer(),
            FakeModel(),
        ),
    )

    provider = FinBERTSentimentProvider()

    try:
        provider.predict_batch(
            [
                "Valid article",
                "",
            ]
        )
        assert False
    except ValueError as exc:
        assert "empty" in str(exc).lower()