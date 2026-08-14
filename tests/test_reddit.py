import pandas as pd

from src.features.time_windows import (
    filter_information_for_prediction,
)
from src.ingestion.reddit_loader import load_reddit


def test_load_reddit():
    posts = load_reddit(
        "data/raw/reddit_sample.csv"
    )

    assert len(posts) == 7

    first_post = posts[0]

    assert first_post.asset == "TCS"
    assert first_post.exchange == "NSE"
    assert first_post.score == 25
    assert first_post.comments == 6


def test_reddit_cutoff():
    posts = load_reddit(
        "data/raw/reddit_sample.csv"
    )

    prediction_timestamp = pd.Timestamp(
        "2026-08-10 10:15:00"
    ).to_pydatetime()

    eligible = filter_information_for_prediction(
        posts,
        prediction_timestamp,
    )

    tcs_posts = [
        post
        for post in eligible
        if post.asset == "TCS"
    ]

    assert len(tcs_posts) == 2

    timestamps = [
        post.published_at
        for post in tcs_posts
    ]

    assert pd.Timestamp(
        "2026-08-10 10:20:00"
    ).to_pydatetime() not in timestamps