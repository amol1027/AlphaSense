"""Unit tests for the read-only shadow scoring script."""

import math

import pandas as pd

from scripts.score_shadow import _is_stale, score_frame, wilson_ci


def _row(asset, pred, label, correct, stale=False):
    return {
        "asset": asset,
        "prediction_timestamp": "2026-10-06 06:15:00+00:00",
        "logged_at": "2026-10-06T06:31:19.803375+00:00",
        "probability": 0.6,
        "prediction": pred,
        "range_median": 0.005,
        "data_age_days": 0.01,
        "stale": stale,
        "realized_range": 0.006,
        "realized_label": label,
        "correct": correct,
    }


def test_wilson_matches_known_pooled_cut():
    lo, hi = wilson_ci(137, 215)
    assert 0.56 < lo < 0.58
    assert 0.69 < hi < 0.71


def test_wilson_empty_is_nan():
    lo, hi = wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_is_stale_parses_logged_strings():
    assert _is_stale(False) is False
    assert _is_stale(True) is True
    assert _is_stale("False") is False
    assert _is_stale("True") is True


def test_score_frame_counts_and_balanced_accuracy():
    df = pd.DataFrame(
        [
            _row("RELIANCE", 1, 1.0, 1.0),  # TP
            _row("RELIANCE", 0, 0.0, 1.0),  # TN
            _row("RELIANCE", 1, 0.0, 0.0),  # FP
            _row("RELIANCE", 0, 1.0, 0.0),  # FN
            _row("TCS", 1, 1.0, 1.0, stale=True),  # excluded from formal set
            _row("TCS", 1, None, None),  # unscored
        ]
    )
    result = score_frame(df)
    assert result["total"] == 6
    assert result["scored"] == 5
    assert result["unscored"] == 1
    assert result["stale_excluded"] == 1
    rel = result["per_asset"]["RELIANCE"]
    assert rel["n"] == 4
    assert rel["accuracy"] == 0.5
    assert rel["ba"] == 0.5  # TPR=1/2, TNR=1/2
    assert rel["maj_base"] == 0.5
    assert result["per_asset"]["TCS"]["n"] == 0
    assert result["pooled"]["n"] == 4
    assert result["pooled"]["accuracy"] == 0.5
