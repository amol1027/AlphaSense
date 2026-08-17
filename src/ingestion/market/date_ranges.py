from datetime import date, timedelta


def split_date_range(
    from_date: date,
    to_date: date,
    max_days: int = 31,
) -> list[tuple[date, date]]:
    """
    Split an inclusive date range into chunks
    containing at most max_days each.
    """

    if to_date < from_date:
        raise ValueError(
            "to_date must be greater than or equal to from_date."
        )

    if max_days < 1:
        raise ValueError(
            "max_days must be at least 1."
        )

    chunks = []

    current = from_date

    while current <= to_date:
        chunk_end = min(
            current + timedelta(days=max_days - 1),
            to_date,
        )

        chunks.append(
            (current, chunk_end)
        )

        current = chunk_end + timedelta(days=1)

    return chunks