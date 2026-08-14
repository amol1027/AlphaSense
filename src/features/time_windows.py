from datetime import datetime, timedelta


def get_prediction_window(
    prediction_timestamp: datetime,
) -> tuple[datetime, datetime]:
    """
    Return the start and end of a one-hour prediction window.

    The prediction timestamp is the beginning of the window.
    """

    window_start = prediction_timestamp
    window_end = prediction_timestamp + timedelta(hours=1)

    return window_start, window_end

def is_information_available(
    information_timestamp: datetime,
    prediction_timestamp: datetime,
) -> bool:
    """
    Return True if information was available at prediction time.
    """

    return information_timestamp <= prediction_timestamp

def filter_news_for_prediction(
    articles,
    prediction_timestamp: datetime,
):
    """
    Return only news articles available at prediction time.
    """

    return [
        article
        for article in articles
        if is_information_available(
            article.published_at,
            prediction_timestamp,
        )
    ]