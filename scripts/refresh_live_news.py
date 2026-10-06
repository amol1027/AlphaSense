"""Refresh the append-only live news cache used by the dashboard.

Polls Upstox + Marketaux + GDELT per asset (best-effort per provider),
deduplicates, and merges into data/interim/live_news.csv with a
7-day retention. Called from POST /refresh and manually:

    python scripts/refresh_live_news.py
"""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("alphasense.refresh_news")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.news.dedup import deduplicate_news  # noqa: E402
from src.ingestion.schemas import NewsArticle  # noqa: E402

CACHE_PATH = ROOT / "data/interim/live_news.csv"
STATUS_PATH = ROOT / "data/interim/live_news_status.json"
RETENTION_DAYS = 7
MAX_PER_ASSET = 200

ASSETS = ("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")

COLUMNS = ["asset", "exchange", "published_at", "source", "headline", "text", "url"]

# Modest limits keep POST /refresh fast. GDELT gets a slightly larger
# allowance since it is keyless and often returns thin matches.
MARKETAUX_LIMIT = 10
UPSTOX_PAGE_SIZE = 10
GDELT_MAX_RECORDS = 25


# Live-refresh budget per provider call. /refresh must stay interactive even
# when an external news API stalls: each call fails fast and is recorded in
# provider_errors instead of holding the refresh lock for minutes.
LIVE_TIMEOUT_SECONDS = 12


def _build_clients() -> tuple[dict[str, object], dict[str, str]]:
    """Instantiate provider clients best-effort; missing tokens => error entry."""
    clients: dict[str, object] = {}
    errors: dict[str, str] = {}

    try:
        from src.ingestion.news.gdelt import GDELTNewsClient

        clients["gdelt"] = GDELTNewsClient(
            timeout=LIVE_TIMEOUT_SECONDS, max_retries=1, backoff_seconds=2
        )
    except Exception as exc:  # pragma: no cover - init rarely fails
        errors["gdelt_init"] = str(exc)

    try:
        from src.ingestion.news.marketaux import MarketauxClient

        clients["marketaux"] = MarketauxClient(timeout=LIVE_TIMEOUT_SECONDS)
    except Exception as exc:
        errors["marketaux_init"] = str(exc)

    try:
        from src.ingestion.news.upstox import UpstoxNewsClient

        clients["upstox"] = UpstoxNewsClient(timeout=LIVE_TIMEOUT_SECONDS)
    except Exception as exc:
        errors["upstox_init"] = str(exc)

    return clients, errors


def _fetch_for_asset(asset: str, clients: dict[str, object]) -> tuple[list[NewsArticle], dict[str, str]]:
    articles: list[NewsArticle] = []
    errors: dict[str, str] = {}

    if "marketaux" in clients:
        try:
            articles.extend(clients["marketaux"].fetch_news(asset, limit=MARKETAUX_LIMIT))  # type: ignore[attr-defined]
        except Exception as exc:
            errors[f"marketaux:{asset}"] = str(exc)
    if "upstox" in clients:
        try:
            articles.extend(clients["upstox"].fetch_news(asset, page_size=UPSTOX_PAGE_SIZE))  # type: ignore[attr-defined]
        except Exception as exc:
            errors[f"upstox:{asset}"] = str(exc)
    if "gdelt" in clients:
        try:
            articles.extend(clients["gdelt"].fetch_news(asset, max_records=GDELT_MAX_RECORDS))  # type: ignore[attr-defined]
        except Exception as exc:
            errors[f"gdelt:{asset}"] = str(exc)

    result = deduplicate_news(articles)
    return result.articles, errors


def _load_existing() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(CACHE_PATH)
    except Exception:
        logger.warning("Live news cache unreadable; starting fresh")
        return pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return df.dropna(subset=["published_at", "url", "asset"])[COLUMNS]


def refresh() -> dict:
    now = pd.Timestamp.now(tz="UTC")
    clients, init_errors = _build_clients()
    provider_errors: dict[str, str] = dict(init_errors)

    fresh_rows: list[dict] = []
    fresh_count = 0
    # I/O-bound provider calls run one asset per thread so a single stalled
    # provider delays the refresh once, not once per asset. Results are merged
    # back in ASSETS order to keep the cache deterministic.
    with ThreadPoolExecutor(max_workers=len(ASSETS)) as pool:
        fetched = list(pool.map(lambda asset: _fetch_for_asset(asset, clients), ASSETS))
    for articles, errors in fetched:
        provider_errors.update(errors)
        fresh_count += len(articles)
        for article in articles:
            fresh_rows.append(
                {
                    "asset": article.asset,
                    "exchange": article.exchange,
                    "published_at": article.published_at,
                    "source": str(article.source),
                    "headline": str(article.headline),
                    "text": str(article.text or ""),
                    "url": str(article.url),
                }
            )

    existing = _load_existing()
    if fresh_rows:
        fresh = pd.DataFrame(fresh_rows, columns=COLUMNS)
        fresh["published_at"] = pd.to_datetime(fresh["published_at"], utc=True, errors="coerce")
        fresh = fresh.dropna(subset=["published_at", "url"])
        combined = pd.concat([existing, fresh], ignore_index=True)
    else:
        combined = existing

    if not combined.empty:
        # Normalize URL for dedup key (trailing slash / case); keep newest row.
        combined["_url_key"] = combined["url"].astype(str).str.strip().str.rstrip("/").str.lower()
        combined = combined.sort_values("published_at").drop_duplicates(["asset", "_url_key"], keep="last")
        cutoff = now - pd.Timedelta(days=RETENTION_DAYS)
        combined = combined[combined["published_at"] >= cutoff]
        # Cap per asset to bound file growth; keep newest.
        combined = (
            combined.sort_values("published_at")
            .groupby("asset", as_index=False)
            .tail(MAX_PER_ASSET)
            .sort_values(["asset", "published_at"])
            .reset_index(drop=True)
        )
        combined = combined.drop(columns=["_url_key"], errors="ignore")
    combined = combined[COLUMNS] if not combined.empty else pd.DataFrame(columns=COLUMNS)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(CACHE_PATH)

    counts = (
        combined["asset"].value_counts().to_dict() if not combined.empty else {asset: 0 for asset in ASSETS}
    )
    for asset in ASSETS:
        counts.setdefault(asset, 0)
    status = {
        "last_refresh_at": datetime.now(timezone.utc).isoformat(),
        "fresh_fetched": int(fresh_count),
        "cache_rows": int(len(combined)),
        "counts_by_asset": {k: int(v) for k, v in counts.items()},
        "provider_errors": provider_errors,
    }
    status_tmp = STATUS_PATH.with_suffix(".tmp")
    status_tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    status_tmp.replace(STATUS_PATH)

    logger.info(
        "Live news cache: %s rows (%s fresh), errors=%d",
        len(combined),
        fresh_count,
        len(provider_errors),
    )
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    summary = refresh()
    print(json.dumps(summary, indent=2))
