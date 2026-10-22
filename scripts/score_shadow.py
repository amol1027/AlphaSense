"""Formal prospective score for the frozen volatility shadow log (read-only).

Usage:
  python scripts/score_shadow.py

Reads data/interim/live_log.csv written by scripts/log_live_prediction.py
and reports the stale-filtered, per-asset cut with Wilson 95% CIs and
balanced accuracy. Never touches the model, artifacts, or locked data.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("alphasense.shadow_score")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_PATH = PROJECT_ROOT / "data/interim/live_log.csv"
ASSETS = ("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return ((center - margin) / denom, (center + margin) / denom)


def _is_stale(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def score_frame(df: pd.DataFrame) -> dict:
    """Score an in-memory shadow-log frame (pure; no I/O)."""
    total = len(df)
    scored = df[pd.notna(df["correct"])].copy() if total else df.copy()
    stale_excluded = int(scored["stale"].map(_is_stale).sum()) if len(scored) else 0
    formal = scored[~scored["stale"].map(_is_stale)].copy() if len(scored) else scored

    per_asset: dict[str, dict] = {}
    for asset in ASSETS:
        sub = formal[formal["asset"] == asset]
        n = len(sub)
        k = int((sub["correct"].astype(float) == 1.0).sum()) if n else 0
        lo, hi = wilson_ci(k, n)
        pred = sub["prediction"].astype(str)
        label = sub["realized_label"].astype(float)
        tp = int(((pred == "1") & (label == 1.0)).sum()) if n else 0
        tn = int(((pred == "0") & (label == 0.0)).sum()) if n else 0
        fp = int(((pred == "1") & (label == 0.0)).sum()) if n else 0
        fn = int(((pred == "0") & (label == 1.0)).sum()) if n else 0
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) else float("nan")
        ba = (tpr + tnr) / 2.0 if n else float("nan")
        maj_base = max(tp + fn, tn + fp) / n if n else float("nan")
        per_asset[asset] = {
            "n": n,
            "accuracy": (k / n) if n else float("nan"),
            "ci_lo": lo,
            "ci_hi": hi,
            "ba": ba,
            "tpr": tpr,
            "tnr": tnr,
            "maj_base": maj_base,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    n_all = len(formal)
    k_all = int((formal["correct"].astype(float) == 1.0).sum()) if n_all else 0
    pooled_lo, pooled_hi = wilson_ci(k_all, n_all)
    return {
        "total": total,
        "scored": len(scored),
        "unscored": total - len(scored),
        "stale_excluded": stale_excluded,
        "per_asset": per_asset,
        "pooled": {
            "n": n_all,
            "accuracy": (k_all / n_all) if n_all else float("nan"),
            "ci_lo": pooled_lo,
            "ci_hi": pooled_hi,
        },
    }


def cmd_score() -> dict:
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"Shadow log is missing at {LOG_PATH}; run scripts/log_live_prediction.py first."
        )
    df = pd.read_csv(LOG_PATH)
    result = score_frame(df)
    logger.info(
        "Shadow: %d scored / %d total (%d unscored, %d stale excluded).",
        result["scored"],
        result["total"],
        result["unscored"],
        result["stale_excluded"],
    )
    logger.info(
        "Pooled accuracy: %.3f [%0.3f, %0.3f] over n=%d (vs 0.50 majority; check BA per asset).",
        result["pooled"]["accuracy"],
        result["pooled"]["ci_lo"],
        result["pooled"]["ci_hi"],
        result["pooled"]["n"],
    )
    for asset in ASSETS:
        row = result["per_asset"][asset]
        logger.info(
            "%s: n=%d acc=%.3f [%.3f, %.3f] BA=%.3f (TPR=%.2f TNR=%.2f) majBase=%.3f",
            asset,
            row["n"],
            row["accuracy"],
            row["ci_lo"],
            row["ci_hi"],
            row["ba"],
            row["tpr"],
            row["tnr"],
            row["maj_base"],
        )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cmd_score()


if __name__ == "__main__":
    main()
