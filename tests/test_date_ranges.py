from datetime import date

import pytest

from src.ingestion.market.date_ranges import (
    split_date_range,
)


def test_single_chunk():
    result = split_date_range(
        date(2026, 8, 1),
        date(2026, 8, 10),
    )

    assert result == [
        (
            date(2026, 8, 1),
            date(2026, 8, 10),
        )
    ]


def test_range_is_split():
    result = split_date_range(
        date(2026, 8, 1),
        date(2026, 8, 10),
        max_days=3,
    )

    assert result == [
        (
            date(2026, 8, 1),
            date(2026, 8, 3),
        ),
        (
            date(2026, 8, 4),
            date(2026, 8, 6),
        ),
        (
            date(2026, 8, 7),
            date(2026, 8, 9),
        ),
        (
            date(2026, 8, 10),
            date(2026, 8, 10),
        ),
    ]


def test_single_day_range():
    result = split_date_range(
        date(2026, 8, 10),
        date(2026, 8, 10),
    )

    assert result == [
        (
            date(2026, 8, 10),
            date(2026, 8, 10),
        )
    ]


def test_invalid_date_range():
    with pytest.raises(
        ValueError,
        match="to_date",
    ):
        split_date_range(
            date(2026, 8, 11),
            date(2026, 8, 10),
        )


def test_invalid_max_days():
    with pytest.raises(
        ValueError,
        match="max_days",
    ):
        split_date_range(
            date(2026, 8, 1),
            date(2026, 8, 10),
            max_days=0,
        )

def test_split_date_range_by_month():
    from src.ingestion.market.date_ranges import (
        split_date_range_by_month,
    )

    result = split_date_range_by_month(
        date(2024, 1, 15),
        date(2024, 3, 10),
    )

    assert result == [
        (
            date(2024, 1, 15),
            date(2024, 1, 31),
        ),
        (
            date(2024, 2, 1),
            date(2024, 2, 29),
        ),
        (
            date(2024, 3, 1),
            date(2024, 3, 10),
        ),
    ]


def test_split_date_range_by_month_single_month():
    from src.ingestion.market.date_ranges import (
        split_date_range_by_month,
    )

    result = split_date_range_by_month(
        date(2026, 8, 1),
        date(2026, 8, 21),
    )

    assert result == [
        (
            date(2026, 8, 1),
            date(2026, 8, 21),
        )
    ]


def test_split_date_range_by_month_leap_year():
    from src.ingestion.market.date_ranges import (
        split_date_range_by_month,
    )

    result = split_date_range_by_month(
        date(2024, 2, 1),
        date(2024, 3, 1),
    )

    assert result == [
        (
            date(2024, 2, 1),
            date(2024, 2, 29),
        ),
        (
            date(2024, 3, 1),
            date(2024, 3, 1),
        ),
    ]


def test_split_date_range_by_month_invalid_range():
    from src.ingestion.market.date_ranges import (
        split_date_range_by_month,
    )

    with pytest.raises(
        ValueError,
        match="to_date",
    ):
        split_date_range_by_month(
            date(2026, 8, 11),
            date(2026, 8, 10),
        )