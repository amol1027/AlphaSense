from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = {
    "asset",
    "exchange",
    "published_at",
    "source",
    "text",
    "positive_probability",
    "neutral_probability",
    "negative_probability",
    "sentiment_score",
}

VALID_ASSETS = {"RELIANCE", "TCS"}
VALID_EXCHANGES = {"NSE"}

PROBABILITY_TOLERANCE = 1e-6
SCORE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class SentimentAuditResult:
    rows: int
    invalid_timestamps: int
    null_timestamps: int
    null_text: int
    empty_text: int
    invalid_assets: int
    invalid_exchanges: int
    invalid_probabilities: int
    probability_sum_violations: int
    score_violations: int
    duplicate_records: int
    future_timestamps: int
    passed: bool


def audit_sentiment_data(
    path: str,
) -> SentimentAuditResult:
    df = pd.read_csv(path)

    missing_columns = (
        REQUIRED_COLUMNS - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required sentiment columns: "
            f"{sorted(missing_columns)}"
        )

    rows = len(df)

    parsed = pd.to_datetime(
        df["published_at"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamps = int(
        parsed.isna().sum()
    )

    null_timestamps = int(
        df["published_at"].isna().sum()
    )

    null_text = int(
        df["text"].isna().sum()
    )

    empty_text = int(
        (
            df["text"]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        ).sum()
    )

    invalid_assets = int(
        (~df["asset"].isin(VALID_ASSETS)).sum()
    )

    invalid_exchanges = int(
        (~df["exchange"].isin(VALID_EXCHANGES)).sum()
    )

    probability_columns = [
        "positive_probability",
        "neutral_probability",
        "negative_probability",
    ]

    probabilities = df[
        probability_columns
    ].apply(pd.to_numeric, errors="coerce")

    invalid_probabilities = int(
        (
            probabilities.isna().any(axis=1)
            | (probabilities < 0).any(axis=1)
            | (probabilities > 1).any(axis=1)
        ).sum()
    )

    probability_sum_violations = int(
        (
            (
                probabilities.sum(axis=1) - 1.0
            ).abs()
            > PROBABILITY_TOLERANCE
        ).sum()
    )

    sentiment_score = pd.to_numeric(
        df["sentiment_score"],
        errors="coerce",
    )

    expected_score = (
        probabilities[
            "positive_probability"
        ]
        - probabilities[
            "negative_probability"
        ]
    )

    score_violations = int(
        (
            (
                sentiment_score
                - expected_score
            ).abs()
            > SCORE_TOLERANCE
        ).sum()
    )

    duplicate_records = int(
        df.duplicated(
            subset=[
                "asset",
                "published_at",
                "source",
                "text",
            ]
        ).sum()
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    future_timestamps = int(
        (
            parsed.notna()
            & (parsed > now)
        ).sum()
    )

    passed = all(
        value == 0
        for value in [
            invalid_timestamps,
            null_timestamps,
            null_text,
            empty_text,
            invalid_assets,
            invalid_exchanges,
            invalid_probabilities,
            probability_sum_violations,
            score_violations,
            duplicate_records,
            future_timestamps,
        ]
    )

    return SentimentAuditResult(
        rows=rows,
        invalid_timestamps=invalid_timestamps,
        null_timestamps=null_timestamps,
        null_text=null_text,
        empty_text=empty_text,
        invalid_assets=invalid_assets,
        invalid_exchanges=invalid_exchanges,
        invalid_probabilities=invalid_probabilities,
        probability_sum_violations=(
            probability_sum_violations
        ),
        score_violations=score_violations,
        duplicate_records=duplicate_records,
        future_timestamps=future_timestamps,
        passed=passed,
    )