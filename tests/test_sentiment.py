from src.sentiment.dummy import DummySentimentProvider


def test_positive_sentiment():
    provider = DummySentimentProvider()

    result = provider.predict(
        "The company reported strong growth and beats expectations."
    )

    assert result.positive_probability == 0.80
    assert result.neutral_probability == 0.15
    assert result.negative_probability == 0.05
    assert result.sentiment_score == 0.75


def test_negative_sentiment():
    provider = DummySentimentProvider()

    result = provider.predict(
        "The company faces regulatory pressure and supply concerns."
    )

    assert result.positive_probability == 0.05
    assert result.neutral_probability == 0.15
    assert result.negative_probability == 0.80
    assert result.sentiment_score == -0.75


def test_neutral_sentiment():
    provider = DummySentimentProvider()

    result = provider.predict(
        "The company announced a routine meeting."
    )

    assert result.positive_probability == 0.10
    assert result.neutral_probability == 0.80
    assert result.negative_probability == 0.10
    assert result.sentiment_score == 0.0