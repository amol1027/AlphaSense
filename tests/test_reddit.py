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

def test_load_reddit_normalizes_timestamp_to_utc(tmp_path):
    path = tmp_path / "reddit.csv"

    pd.DataFrame(
        {
            "asset": ["TCS"],
            "exchange": ["NSE"],
            "published_at": ["2026-08-10 09:20:00"],
            "source": ["reddit"],
            "headline": ["Test"],
            "text": ["Test post"],
            "url": ["https://reddit.com/test"],
            "score": [10],
            "comments": [2],
        }
    ).to_csv(path, index=False)

    result = load_reddit(path)

    assert result[0].published_at.tzinfo is not None
    assert result[0].published_at.utcoffset() is not None