from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.ingestion.schemas import NewsArticle


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}


@dataclass(frozen=True)
class DeduplicatedNews:
    articles: list[NewsArticle]
    duplicate_count: int


def normalize_url(url: str) -> str:
    parts = urlsplit(str(url))

    query = [
        (key, value)
        for key, value in parse_qsl(
            parts.query,
            keep_blank_values=True,
        )
        if key.lower() not in TRACKING_PARAMS
    ]

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            urlencode(query),
            "",
        )
    )


def normalize_headline(headline: str) -> str:
    return " ".join(
        str(headline)
        .lower()
        .split()
    )


def deduplicate_news(
    articles: list[NewsArticle],
    time_tolerance: timedelta = timedelta(
        minutes=10
    ),
) -> DeduplicatedNews:
    """
    Deduplicate news conservatively.

    Priority:
        1. Exact normalized URL.
        2. Same normalized headline + same asset
           within the time tolerance.

    The first occurrence is retained.
    """

    unique: list[NewsArticle] = []
    seen_urls: set[str] = set()

    seen_headlines: dict[
        tuple[str, str],
        list,
    ] = {}

    duplicate_count = 0

    for article in articles:
        url_key = normalize_url(
            str(article.url)
        )

        if url_key in seen_urls:
            duplicate_count += 1
            continue

        headline_key = (
            article.asset,
            normalize_headline(
                article.headline
            ),
        )

        is_duplicate = False

        for existing_time in seen_headlines.get(
            headline_key,
            [],
        ):
            if abs(
                article.published_at
                - existing_time
            ) <= time_tolerance:
                is_duplicate = True
                break

        if is_duplicate:
            duplicate_count += 1
            continue

        seen_urls.add(url_key)

        seen_headlines.setdefault(
            headline_key,
            [],
        ).append(article.published_at)

        unique.append(article)

    return DeduplicatedNews(
        articles=unique,
        duplicate_count=duplicate_count,
    )