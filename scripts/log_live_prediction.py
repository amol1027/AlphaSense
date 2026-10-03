"""Live paper-trading log for the frozen volatility model (read-only vs model).

Usage:
  log today's predictions:   python scripts/log_live_prediction.py
  fill realized outcomes:     python scripts/log_live_prediction.py --label

Log: data/interim/live_log.csv keyed by (asset, prediction_timestamp);
re-runs never duplicate. Realized labels use each artifact's frozen
range_median (same threshold the server predicts against). The model,
artifacts, and locked data are never touched.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("alphasense.shadow_log")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_PATH = PROJECT_ROOT / "data/interim/live_log.csv"
ASSETS = ("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")
MAX_BAR_AGE = pd.Timedelta(minutes=45)


def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        df["prediction_timestamp"] = pd.to_datetime(
            df["prediction_timestamp"], utc=True, format="mixed"
        )
        return df
    return pd.DataFrame(
        columns=["asset", "prediction_timestamp", "logged_at", "probability",
                 "prediction", "range_median", "data_age_days", "stale",
                 "realized_range", "realized_label", "correct"]
    )


def cmd_log() -> int:
    from src.prediction.server import latest_bars
    from src.prediction.service import predict_range_regime

    log = load_log()
    now = pd.Timestamp.now(tz="UTC")
    added = 0
    for asset in ASSETS:
        try:
            bars = latest_bars(asset)
            res = predict_range_regime(asset, bars)
        except ValueError as exc:
            logger.info("%s: skipped (%s)", asset, exc)
            continue
        age = now - pd.Timestamp(res["prediction_timestamp"])
        if age < pd.Timedelta(0) or age > MAX_BAR_AGE:
            logger.info("%s: skipped; latest bar is %s old", asset, age)
            continue
        key = pd.Timestamp(res["prediction_timestamp"])
        key = key.tz_localize("UTC") if key.tzinfo is None else key.tz_convert("UTC")
        dup = (
            (log["asset"] == asset)
            & (pd.to_datetime(log["prediction_timestamp"], utc=True, format="mixed") == key)
        ).any() if not log.empty else False
        if dup:
            logger.info("%s: already logged for %s", asset, res['prediction_timestamp'])
            continue
        log.loc[len(log)] = {
            "asset": asset,
            "prediction_timestamp": res["prediction_timestamp"],
            "logged_at": now.isoformat(),
            "probability": res["probability_high_range"],
            "prediction": res["prediction"],
            "range_median": res["range_median"],
            "data_age_days": res["data_age_days"],
            "stale": res["stale"],
            "realized_range": pd.NA,
            "realized_label": pd.NA,
            "correct": pd.NA,
        }
        added += 1
        logger.info("%s: logged p=%s pred=%s", asset, res['probability_high_range'], res['prediction'])
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG_PATH, index=False)
    logger.info("Logged %d new predictions -> %s", added, LOG_PATH)
    return added


def cmd_label() -> int:
    log = load_log()
    if log.empty:
        logger.info("Log is empty; run without --label first.")
        return 0
    market_path = PROJECT_ROOT / "data/interim/live_market_15m.csv"
    if not market_path.exists():
        raise FileNotFoundError(
            "Live bar cache is missing; run scripts/refresh_live_market.py first."
        )
    market = pd.read_csv(market_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    labeled = 0
    for i, row in log.iterrows():
        if pd.notna(row["realized_label"]):
            continue
        t = pd.Timestamp(row["prediction_timestamp"])
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        window = market[
            (market["asset"] == row["asset"])
            & (market["timestamp"] > t)
            & (market["timestamp"] <= t + pd.Timedelta(hours=1))
        ]
        if len(window) < 4:
            continue
        prior = market[
            (market["asset"] == row["asset"]) & (market["timestamp"] <= t)
        ].sort_values("timestamp")
        if prior.empty:
            continue
        close_at_t = prior.iloc[-1]["close"]
        realized = (window["high"].max() - window["low"].min()) / close_at_t
        label = int(realized > float(row["range_median"]))
        log.at[i, "realized_range"] = realized
        log.at[i, "realized_label"] = label
        log.at[i, "correct"] = int(label == int(row["prediction"]))
        labeled += 1
    log.to_csv(LOG_PATH, index=False)
    done = log[pd.notna(log["correct"])]
    logger.info("Labeled %d rows (%d total realized).", labeled, len(done))
    if not done.empty:
        logger.info("Live hit rate: %.3f over %d predictions",
                    done['correct'].astype(int).mean(), len(done))
    return labeled


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "--label":
        cmd_label()
    else:
        cmd_log()


if __name__ == "__main__":
    main()
