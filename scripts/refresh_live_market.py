"""Refresh the append-only OHLCV cache used by shadow predictions.

Run during the NSE session before logging predictions. The cache is separate
from the frozen research dataset and includes only completed 15-minute bars.
"""

from pathlib import Path
import logging
import sys
import json
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger("alphasense.refresh")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.market.upstox import fetch_intraday_candles

CACHE_PATH = ROOT / "data/interim/live_market_15m.csv"
STATUS_PATH = ROOT / "data/interim/live_status.json"
RETENTION_DAYS = 7
ASSETS = {
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "INFY": "NSE_EQ|INE009A01021",
    "ICICIBANK": "NSE_EQ|INE090A01021",
}
COLUMNS = ["asset", "exchange", "timestamp", "open", "high", "low", "close", "volume"]
INTERVAL = pd.Timedelta(minutes=15)


def _seed_history() -> pd.DataFrame:
    frames = []
    for asset in ASSETS:
        path = ROOT / f"data/raw/market/phase1_{asset.lower()}_15m.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing history needed for feature warmup: {path}")
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frames.append(frame[COLUMNS])
    return pd.concat(frames, ignore_index=True)


def refresh() -> int:
    now = pd.Timestamp.now(tz="UTC")
    existing = pd.read_csv(CACHE_PATH) if CACHE_PATH.exists() else _seed_history()
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    incoming = []
    for asset, instrument_key in ASSETS.items():
        candles = fetch_intraday_candles(instrument_key, interval_minutes=15)
        if candles.empty:
            logger.warning("%s: no intraday candles returned", asset)
            continue
        candles = candles[candles["timestamp"] + INTERVAL <= now].copy()
        if candles.empty:
            logger.info("%s: no completed 15-minute candles yet", asset)
            continue
        candles["asset"] = asset
        candles["exchange"] = "NSE"
        incoming.append(candles[COLUMNS])

    if incoming:
        combined = pd.concat([existing, *incoming], ignore_index=True)
        cutoff = now - pd.Timedelta(days=RETENTION_DAYS)
        combined = combined[combined["timestamp"] >= cutoff]
    else:
        combined = existing
    combined = combined.drop_duplicates(["asset", "exchange", "timestamp"], keep="last")
    combined = combined.sort_values(["asset", "exchange", "timestamp"]).reset_index(drop=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = CACHE_PATH.with_suffix(".tmp")
    combined.to_csv(temporary_path, index=False)
    temporary_path.replace(CACHE_PATH)
    latest_by_asset = {
        asset: combined.loc[combined["asset"] == asset, "timestamp"].max().isoformat()
        for asset in ASSETS
        if (combined["asset"] == asset).any()
    }
    status = {
        "last_refresh_at": datetime.now(timezone.utc).isoformat(),
        "completed_candles_received": int(sum(len(frame) for frame in incoming)),
        "cache_rows": int(len(combined)),
        "latest_bar_by_asset": latest_by_asset,
    }
    status_tmp = STATUS_PATH.with_suffix(".tmp")
    status_tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    status_tmp.replace(STATUS_PATH)
    logger.info(
        "Updated %s: %s bars retained (rolling %s days, %s new frames)",
        CACHE_PATH,
        f"{len(combined):,}",
        RETENTION_DAYS,
        len(incoming),
    )
    return len(incoming)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    refresh()
