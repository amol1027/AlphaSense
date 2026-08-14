from src.ingestion.loader import load_news


def test_load_news():
    articles = load_news("data/raw/news_sample.csv")

    assert len(articles) == 5

    first_article = articles[0]

    assert first_article.asset == "AAPL"
    assert first_article.source == "Example News"
    assert first_article.headline == "Apple reports strong quarterly growth"
    assert first_article.text != ""
    assert str(first_article.url) == "https://example.com/1"