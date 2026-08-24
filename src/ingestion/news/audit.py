from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = {
    "asset",
    "exchange",
    "published_at",
    "source",
    "headline",
    "text",
    "url",
}

VALID_ASSETS = {
    "TCS",
    "RELIANCE",
}

VALID_EXCHANGES = {
    "NSE",
}


@dataclass
class NewsAuditResult:
    rows: int

    assets: set[str]
    sources: dict[str, int]

    start_timestamp: pd.Timestamp | None
    end_timestamp: pd.Timestamp | None

    invalid_timestamps: int = 0
    null_timestamps: int = 0

    null_headlines: int = 0
    empty_headlines: int = 0
    empty_bodies: int = 0

    duplicate_urls: int = 0
    duplicate_headlines: int = 0

    invalid_assets: int = 0
    invalid_exchanges: int = 0

    future_timestamps: int = 0
    out_of_window_timestamps: int = 0

    passed: bool = True

    @property
    def status(self) -> str:
        return (
            "PASS"
            if self.passed
            else "FAIL"
        )


def _normalize_headline(
    value: object,
) -> str:
    return " ".join(
        str(value)
        .strip()
        .lower()
        .split()
    )


def _provider_name(
    source: object,
) -> str:
    value = str(source)

    if ":" in value:
        return value.split(
            ":",
            1,
        )[0].lower()

    return value.lower()


def audit_news_data(
    df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> NewsAuditResult:
    """
    Audit the Phase 1 historical news dataset.

    The function does not modify the input DataFrame.

    Empty article bodies are reported but are not
    considered a failure because sentiment preparation
    can fall back to the headline.
    """

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    timestamps = pd.to_datetime(
        df["published_at"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = int(
        timestamps.isna().sum()
    )

    null_timestamps = int(
        df["published_at"].isna().sum()
    )

    valid_timestamps = timestamps.dropna()

    start_timestamp = (
        valid_timestamps.min()
        if not valid_timestamps.empty
        else None
    )

    end_timestamp = (
        valid_timestamps.max()
        if not valid_timestamps.empty
        else None
    )

    headlines = (
        df["headline"]
        .fillna("")
        .astype(str)
    )

    bodies = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    urls = (
        df["url"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_headlines = int(
        headlines.str.strip().eq("").sum()
    )

    null_headlines = int(
        df["headline"].isna().sum()
    )

    empty_bodies = int(
        bodies.str.strip().eq("").sum()
    )

    duplicate_urls = int(
        urls[
            urls.ne("")
        ].duplicated().sum()
    )

    normalized_headlines = (
        headlines.map(_normalize_headline)
    )

    duplicate_headlines = int(
        normalized_headlines[
            normalized_headlines.ne("")
        ].duplicated().sum()
    )

    invalid_assets = int(
        (~df["asset"].isin(VALID_ASSETS))
        .sum()
    )

    invalid_exchanges = int(
        (~df["exchange"].isin(VALID_EXCHANGES))
        .sum()
    )

    future_timestamps = 0

    if end_date is not None:
        end_date = pd.Timestamp(
            end_date
        )

        if end_date.tzinfo is None:
            end_date = end_date.tz_localize(
                "UTC"
            )
        else:
            end_date = end_date.tz_convert(
                "UTC"
            )

        future_timestamps = int(
            (
                valid_timestamps
                >= end_date
            ).sum()
        )

    out_of_window_timestamps = 0

    if (
        start_date is not None
        or end_date is not None
    ):
        window_mask = pd.Series(
            False,
            index=df.index,
        )

        valid_mask = timestamps.notna()

        if start_date is not None:
            start_date = pd.Timestamp(
                start_date
            )

            if start_date.tzinfo is None:
                start_date = (
                    start_date.tz_localize(
                        "UTC"
                    )
                )
            else:
                start_date = (
                    start_date.tz_convert(
                        "UTC"
                    )
                )

            window_mask |= (
                valid_mask
                & (timestamps < start_date)
            )

        if end_date is not None:
            window_mask |= (
                valid_mask
                & (timestamps >= end_date)
            )

        out_of_window_timestamps = int(
            window_mask.sum()
        )

    assets = set(
        df["asset"]
        .dropna()
        .astype(str)
    )

    sources = (
        df["source"]
        .fillna("")
        .map(_provider_name)
        .value_counts()
        .to_dict()
    )

    passed = all(
        [
            invalid_timestamps == 0,
            duplicate_urls == 0,
            empty_headlines == 0,
            invalid_assets == 0,
            invalid_exchanges == 0,
            future_timestamps == 0,
            out_of_window_timestamps == 0,
        ]
    )

    return NewsAuditResult(
        rows=len(df),
        assets=assets,
        sources=sources,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        invalid_timestamps=invalid_timestamps,
        null_timestamps=null_timestamps,
        null_headlines=null_headlines,
        empty_headlines=empty_headlines,
        empty_bodies=empty_bodies,
        duplicate_urls=duplicate_urls,
        duplicate_headlines=duplicate_headlines,
        invalid_assets=invalid_assets,
        invalid_exchanges=invalid_exchanges,
        future_timestamps=future_timestamps,
        out_of_window_timestamps=(
            out_of_window_timestamps
        ),
        passed=passed,
    )