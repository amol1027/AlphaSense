from pathlib import Path

from src.sentiment.audit import (
    audit_sentiment_data,
)


INPUT_PATH = Path(
    "data/processed/research_news_sentiment.csv"
)


def main():
    print("=" * 70)
    print("PHASE 1 SENTIMENT DATA AUDIT")
    print("=" * 70)
    print(f"Input: {INPUT_PATH}")
    print()

    result = audit_sentiment_data(
        str(INPUT_PATH)
    )

    print("DATA QUALITY")
    print("-" * 70)
    print(
        f"Rows:                       "
        f"{result.rows:,}"
    )
    print(
        f"Invalid timestamps:         "
        f"{result.invalid_timestamps}"
    )
    print(
        f"Null timestamps:            "
        f"{result.null_timestamps}"
    )
    print(
        f"Null text:                  "
        f"{result.null_text}"
    )
    print(
        f"Empty text:                 "
        f"{result.empty_text}"
    )
    print(
        f"Invalid assets:             "
        f"{result.invalid_assets}"
    )
    print(
        f"Invalid exchanges:          "
        f"{result.invalid_exchanges}"
    )
    print(
        f"Invalid probabilities:      "
        f"{result.invalid_probabilities}"
    )
    print(
        f"Probability sum violations:"
        f" {result.probability_sum_violations}"
    )
    print(
        f"Score violations:           "
        f"{result.score_violations}"
    )
    print(
        f"Duplicate records:          "
        f"{result.duplicate_records}"
    )
    print(
        f"Future timestamps:          "
        f"{result.future_timestamps}"
    )
    print()

    print(
        "OVERALL STATUS: "
        + ("PASS" if result.passed else "FAIL")
    )

    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()