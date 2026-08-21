from datetime import datetime, timedelta

import pandas as pd


def get_prediction_window(
    prediction_timestamp: datetime,
) -> tuple[datetime, datetime]:
    """
    Return the start and end of a one-hour prediction window.

    The prediction timestamp is the beginning of the window.
    """

    window_start = prediction_timestamp
    window_end = (
        prediction_timestamp
        + timedelta(hours=1)
    )

    return window_start, window_end


def is_information_available(
    information_timestamp: datetime,
    prediction_timestamp: datetime,
) -> bool:
    """
    Return True if information was available at
    prediction time.

    Timestamps are normalized to UTC before comparison.
    """

    information_timestamp = pd.Timestamp(
        information_timestamp
    )

    prediction_timestamp = pd.Timestamp(
        prediction_timestamp
    )

    if information_timestamp.tzinfo is None:
        information_timestamp = (
            information_timestamp.tz_localize(
                "UTC"
            )
        )
    else:
        information_timestamp = (
            information_timestamp.tz_convert(
                "UTC"
            )
        )

    if prediction_timestamp.tzinfo is None:
        prediction_timestamp = (
            prediction_timestamp.tz_localize(
                "UTC"
            )
        )
    else:
        prediction_timestamp = (
            prediction_timestamp.tz_convert(
                "UTC"
            )
        )

    return (
        information_timestamp
        <= prediction_timestamp
    )


def filter_news_for_prediction(
    articles,
    prediction_timestamp: datetime,
    lookback: timedelta = timedelta(hours=1),
):
    """
    Return news published during the lookback window
    ending at prediction time.

    The window is:

        prediction_timestamp - lookback
            <= published_at
            <= prediction_timestamp

    This prevents future information leakage while
    avoiding accumulation of all historical news.
    """

    prediction_timestamp = pd.Timestamp(
        prediction_timestamp
    )

    if prediction_timestamp.tzinfo is None:
        prediction_timestamp = (
            prediction_timestamp.tz_localize(
                "UTC"
            )
        )
    else:
        prediction_timestamp = (
            prediction_timestamp.tz_convert(
                "UTC"
            )
        )

    window_start = (
        prediction_timestamp
        - lookback
    )

    filtered = []

    for article in articles:

        published_at = pd.Timestamp(
            article.published_at
        )

        if published_at.tzinfo is None:
            published_at = (
                published_at.tz_localize(
                    "UTC"
                )
            )
        else:
            published_at = (
                published_at.tz_convert(
                    "UTC"
                )
            )

        if (
            window_start
            <= published_at
            <= prediction_timestamp
        ):
            filtered.append(article)

    return filtered


def filter_information_for_prediction(
    items,
    prediction_timestamp: datetime,
):
    """
    Return information that was available at
    prediction time.

    This remains a cumulative availability filter.

    It is used by existing Reddit feature logic.
    """

    return [
        item
        for item in items
        if is_information_available(
            item.published_at,
            prediction_timestamp,
        )
    ]