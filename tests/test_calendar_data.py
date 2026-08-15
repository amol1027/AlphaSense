import json
from datetime import date
from pathlib import Path


CALENDAR_PATH = Path(
    "data/reference/nse_bse_holidays_2026.json"
)


def test_2026_calendar_file_exists():
    assert CALENDAR_PATH.exists()


def test_2026_calendar_has_expected_holidays():
    with CALENDAR_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    holidays = {
        date.fromisoformat(value)
        for value in data["holidays"]
    }

    assert data["year"] == 2026
    assert data["market"] == "equity_normal"

    assert date(2026, 1, 15) in holidays
    assert date(2026, 1, 26) in holidays
    assert date(2026, 5, 28) in holidays
    assert date(2026, 12, 25) in holidays

    assert len(holidays) == 16