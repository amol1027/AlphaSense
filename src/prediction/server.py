"""Local prediction HTTP server (JSON API + React webapp host).

API endpoints (JSON):
  GET  /health              -> server and feed status
  GET  /dashboard           -> feed, five predictions, and recent outcomes
  GET  /chart               -> intraday OHLC history for one stock
  GET  /stream              -> server-side events relayed from Upstox V3
  GET  /news?asset=XXX      -> latest cached news headlines for one stock
  GET  /predict?asset=XXX   -> prediction from latest local 15m bars
  POST /refresh             -> fetch candles + news, update shadow predictions and score outcomes
  POST /predict             -> {"asset": ..., "bars": [{asset,exchange,timestamp,open,high,low,close,volume}]}

UI routes (/, /markets, /login, /register, /assets/*) serve the production
Vite build from website/dist/ when present, with an SPA fallback to
index.html for extensionless paths. The React website is the only dashboard.
"""

import json
import logging
import os
import queue
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.prediction.artifacts import artifact_path

logger = logging.getLogger("alphasense.prediction")

import pandas as pd

from src.ingestion.market.upstox import fetch_intraday_candles
from src.prediction.live_feed import live_market_feed
from src.prediction.service import predict_range_regime
from scripts.log_live_prediction import cmd_label as label_shadow_outcomes, cmd_log as log_shadow_predictions
from scripts.refresh_live_market import refresh as refresh_market_data
from scripts.refresh_live_news import refresh as refresh_news_data

KNOWN_ASSETS = ("RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK")
INSTRUMENT_KEYS = {
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "INFY": "NSE_EQ|INE009A01021",
    "ICICIBANK": "NSE_EQ|INE090A01021",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Browsers send Origin on POSTs through the Vite dev/preview proxy, where
# Origin (e.g. http://localhost:5173) never equals the backend Host
# (127.0.0.1:8000). Allow same-origin plus the local Vite dev/preview
# origins; everything else is still rejected.
DEV_BROWSER_ORIGINS = frozenset({
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
})
MARKET_CACHE_PATH = PROJECT_ROOT / "data/interim/live_market_15m.csv"
LIVE_LOG_PATH = PROJECT_ROOT / "data/interim/live_log.csv"
STATUS_PATH = PROJECT_ROOT / "data/interim/live_status.json"
NEWS_CACHE_PATH = PROJECT_ROOT / "data/interim/live_news.csv"
NEWS_STATUS_PATH = PROJECT_ROOT / "data/interim/live_news_status.json"
_MARKET_CACHE_SIGNATURE = None
_MARKET_CACHE_DATA = None
_NEWS_CACHE_SIGNATURE = None
_NEWS_CACHE_DATA = None
_REFRESH_LOCK = threading.Lock()

# The React website (website/) is the only dashboard. This service is a JSON
# API; it additionally hosts the production Vite build (website/dist/) so one
# process can serve both API and UI locally. Regenerate dist/ with
# `pnpm --dir website build`. When dist/ is absent, UI routes return a JSON
# pointer instead of HTML.
WEBAPP_DIR = PROJECT_ROOT / "website" / "dist"
WEBAPP_INDEX = WEBAPP_DIR / "index.html"

_STATIC_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


def _webapp_path(request_path: str) -> Path | None:
    """Resolve a URL path to a file under website/dist/, else None."""
    relative = request_path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
    if not relative or ".." in relative.split("/"):
        return None
    candidate = WEBAPP_DIR / relative
    try:
        candidate.resolve().relative_to(WEBAPP_DIR.resolve())
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def load_live_market() -> pd.DataFrame:
    global _MARKET_CACHE_SIGNATURE, _MARKET_CACHE_DATA
    if not MARKET_CACHE_PATH.exists():
        raise ValueError("Live market cache is missing; run scripts/refresh_live_market.py first")
    stat = MARKET_CACHE_PATH.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    if signature != _MARKET_CACHE_SIGNATURE:
        _MARKET_CACHE_DATA = pd.read_csv(MARKET_CACHE_PATH)
        _MARKET_CACHE_DATA["timestamp"] = pd.to_datetime(
            _MARKET_CACHE_DATA["timestamp"], utc=True
        )
        _MARKET_CACHE_SIGNATURE = signature
    return _MARKET_CACHE_DATA


def load_live_news() -> pd.DataFrame:
    global _NEWS_CACHE_SIGNATURE, _NEWS_CACHE_DATA
    if not NEWS_CACHE_PATH.exists():
        return pd.DataFrame(columns=["asset", "exchange", "published_at", "source", "headline", "text", "url"])
    stat = NEWS_CACHE_PATH.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    if signature != _NEWS_CACHE_SIGNATURE:
        _NEWS_CACHE_DATA = pd.read_csv(NEWS_CACHE_PATH)
        if not _NEWS_CACHE_DATA.empty:
            _NEWS_CACHE_DATA["published_at"] = pd.to_datetime(
                _NEWS_CACHE_DATA["published_at"], utc=True, errors="coerce"
            )
        _NEWS_CACHE_SIGNATURE = signature
    return _NEWS_CACHE_DATA


def news_status() -> dict:
    if NEWS_STATUS_PATH.exists():
        try:
            return json.loads(NEWS_STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"last_refresh_at": None, "cache_rows": 0, "counts_by_asset": {}, "provider_errors": {}}


def news_for_asset(asset: str, limit: int = 20, offset: int = 0) -> tuple[list[dict], dict, int]:
    news = load_live_news()
    if news.empty:
        return [], news_status(), 0
    frame = news[news["asset"] == asset].copy()
    if frame.empty:
        return [], news_status(), 0
    frame = frame.sort_values("published_at", ascending=False)
    total = len(frame)
    limit = max(1, min(limit, 50))
    offset = max(0, offset)
    frame = frame.iloc[offset:offset + limit]
    items = [
        {
            "headline": str(row.headline),
            "source": str(row.source),
            "published_at": pd.Timestamp(row.published_at).isoformat(),
            "url": str(row.url),
            "text": str(row.text or "")[:500],
        }
        for row in frame.itertuples(index=False)
    ]
    return items, news_status(), total


def latest_bars(asset: str, n: int = 30, market: pd.DataFrame | None = None) -> pd.DataFrame:
    df = (market if market is not None else load_live_market())
    df = df[df["asset"] == asset]
    if df.empty:
        raise ValueError(f"No cached bars for asset {asset!r}")
    return df.sort_values("timestamp").tail(n).reset_index(drop=True)


def chart_bars(asset: str, interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return intraday OHLC history from Upstox, aggregated to a chart interval."""
    if interval not in {"1m", "5m", "15m", "1h"}:
        raise ValueError(f"unsupported chart interval {interval!r}")
    try:
        minute_source = fetch_intraday_candles(INSTRUMENT_KEYS[asset], interval_minutes=1)
    except Exception:
        minute_source = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if interval == "15m":
        source = load_live_market()
        source = source[source["asset"] == asset].sort_values("timestamp")
        result = source.tail(500).copy()
    else:
        source = minute_source
        if source.empty:
            raise ValueError("No intraday chart data is available for this stock")
        if interval == "1m":
            result = source.tail(500).copy()
        else:
            minutes = 5 if interval == "5m" else 60
            indexed = source.set_index("timestamp")
            grouped = indexed.resample(
                f"{minutes}min", origin="start_day", offset="3h45min",
                label="left", closed="left",
            )
            result = grouped.agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"), count=("close", "count"),
            )
            result = result[result["count"] == minutes].drop(columns="count").reset_index()
            result = result.tail(500)
    if result.empty:
        raise ValueError("No chart data is available for this stock yet")
    return (
        result[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True),
        minute_source[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True),
    )


def feed_status(market: pd.DataFrame | None = None) -> dict:
    refreshed_at = None
    if STATUS_PATH.exists():
        try:
            refreshed_at = json.loads(STATUS_PATH.read_text(encoding="utf-8")).get("last_refresh_at")
        except (OSError, json.JSONDecodeError):
            pass
    if market is None:
        try:
            market = load_live_market()
        except (ValueError, OSError):
            return {
                "state": "waiting",
                "latest_bar_timestamp": None,
                "age_minutes": None,
                "last_refresh_at": refreshed_at,
            }
    if market.empty:
        state, latest, age = "waiting", None, None
    else:
        latest = pd.to_datetime(market["timestamp"], utc=True).max()
        age = max(0.0, (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 60)
        state = "fresh" if age <= 45 else "delayed" if age < 7 * 24 * 60 else "stale"
    return {
        "state": state,
        "latest_bar_timestamp": latest.isoformat() if latest is not None else None,
        "age_minutes": round(age, 1) if age is not None else None,
        "last_refresh_at": refreshed_at,
    }


def recent_predictions(limit: int = 10) -> list[dict]:
    if not LIVE_LOG_PATH.exists():
        return []
    try:
        log = pd.read_csv(LIVE_LOG_PATH)
    except (OSError, pd.errors.ParserError):
        return []
    required = {"asset", "prediction_timestamp", "probability", "prediction", "realized_label", "correct"}
    if log.empty or not required.issubset(log.columns):
        return []
    log["prediction_timestamp"] = pd.to_datetime(log["prediction_timestamp"], utc=True, errors="coerce")
    log = log.dropna(subset=["prediction_timestamp"]).sort_values("prediction_timestamp", ascending=False).head(limit)
    items = []
    now = pd.Timestamp.now(tz="UTC")
    for _, row in log.iterrows():
        prediction = int(row["prediction"])
        probability_high_range = float(row["probability"])
        prediction_time = row["prediction_timestamp"]
        correct = None if pd.isna(row["correct"]) else bool(int(row["correct"]))
        local_time = prediction_time.tz_convert("Asia/Kolkata")
        last_session_bar = local_time.normalize() + pd.Timedelta(hours=15, minutes=15)
        cannot_fit_before_close = (
            local_time.weekday() < 5
            and prediction_time + pd.Timedelta(hours=1) > last_session_bar
        )
        if correct is not None:
            outcome_status = "scored"
        elif cannot_fit_before_close:
            outcome_status = "not_scored"
        elif prediction_time + pd.Timedelta(hours=1) <= now:
            outcome_status = "awaiting_data"
        else:
            outcome_status = "pending"
        items.append({
            "asset": str(row["asset"]),
            "prediction_timestamp": prediction_time.isoformat(),
            "probability": probability_high_range if prediction else 1 - probability_high_range,
            "prediction": prediction,
            "realized_label": None if pd.isna(row["realized_label"]) else int(row["realized_label"]),
            "correct": correct,
            "outcome_status": outcome_status,
        })
    return items


def service_health() -> tuple[int, dict]:
    """Readiness for supervisors/monitors: 200 only when every check passes.

    Token presence (not validity) is reported — no network probe here, since
    health endpoints are polled frequently. Validity is enforced by the
    shadow runner's startup gate and visible via cache freshness."""
    try:
        market = load_live_market()
    except (ValueError, OSError):
        market = None
    feed = feed_status(market)
    cache_ok = market is not None and not market.empty and feed["state"] in ("fresh", "delayed")
    missing = [asset for asset in KNOWN_ASSETS if not artifact_path(asset).is_file()]
    token_ok = bool(os.getenv("UPSTOX_ACCESS_TOKEN"))
    webapp_ok = WEBAPP_INDEX.is_file()
    checks = {
        "market_cache": {"ok": cache_ok, "state": feed["state"], "age_minutes": feed["age_minutes"]},
        "model_artifacts": {"ok": not missing, **({"missing": missing} if missing else {})},
        "upstox_token_configured": {"ok": token_ok},
        "webapp": {"ok": webapp_ok},
    }
    healthy = all(check["ok"] for check in checks.values())
    return (200 if healthy else 503), {
        "status": "ok" if healthy else "degraded",
        "assets": list(KNOWN_ASSETS),
        "feed": feed,
        "checks": checks,
    }


def dashboard_snapshot() -> dict:
    try:
        market = load_live_market()
    except (ValueError, OSError):
        market = None
    predictions = {}
    for asset in KNOWN_ASSETS:
        try:
            if market is None:
                raise ValueError("Live market cache is missing; run scripts/refresh_live_market.py first")
            predictions[asset] = predict_range_regime(asset, latest_bars(asset, market=market))
        except ValueError as exc:
            predictions[asset] = {"error": str(exc)}
    return {
        "feed": feed_status(market),
        "predictions": predictions,
        "history": recent_predictions(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AlphaSenseVol/1.0"
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str, cache_control: str) -> None:
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._send(404, {"error": f"static asset unavailable: {exc}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_webapp_index(self) -> None:
        if WEBAPP_INDEX.is_file():
            self._send_file(WEBAPP_INDEX, "text/html; charset=utf-8", "no-store")
        else:
            self._send(404, {
                "error": "website build not found",
                "hint": "Run `pnpm --dir website build`, or start the Vite dev server and open its /markets page.",
            })

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_webapp_index()
            return
        if parsed.path == "/api":
            self._send(200, {
                "service": "AlphaSense market prediction API",
                "dashboard": "Open the React website and choose Markets.",
            })
            return
        if parsed.path == "/dashboard":
            self._send(200, dashboard_snapshot())
            return
        if parsed.path == "/stream":
            client = live_market_feed.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    try:
                        message = client.get(timeout=15)
                        body = f"data: {json.dumps(message, separators=(',', ':'))}\n\n".encode()
                    except queue.Empty:
                        body = b": keepalive\n\n"
                    self.wfile.write(f"{len(body):X}\r\n".encode() + body + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                live_market_feed.unsubscribe(client)
                try:
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    pass
            return
        if parsed.path == "/chart":
            query = parse_qs(parsed.query)
            asset = query.get("asset", [None])[0]
            interval = query.get("interval", ["15m"])[0]
            if not asset or asset not in KNOWN_ASSETS:
                self._send(400, {"error": f"unknown asset {asset!r}"})
                return
            try:
                bars, minute_bars = chart_bars(asset, interval)
                serialize = lambda frame: [
                    {"time": int(pd.Timestamp(row.timestamp).timestamp()),
                     "open": float(row.open), "high": float(row.high),
                     "low": float(row.low), "close": float(row.close)}
                    for row in frame.itertuples(index=False)
                ]
                self._send(200, {
                    "asset": asset,
                    "interval": interval,
                    "bars": serialize(bars),
                    "minute_bars": serialize(minute_bars),
                })
            except (ValueError, OSError) as exc:
                self._send(200, {"asset": asset, "interval": interval, "bars": [], "message": str(exc)})
            return
        if parsed.path == "/health":
            code, payload = service_health()
            self._send(code, payload)
            return
        if parsed.path == "/news":
            query = parse_qs(parsed.query)
            asset = query.get("asset", [None])[0]
            try:
                limit = int(query.get("limit", ["20"])[0])
            except ValueError:
                limit = 20
            try:
                offset = int(query.get("offset", ["0"])[0])
            except ValueError:
                offset = 0
            limit = max(1, min(limit, 50))
            offset = max(0, offset)
            if not asset or asset not in KNOWN_ASSETS:
                self._send(400, {"error": f"unknown asset {asset!r}"})
                return
            try:
                items, status, total = news_for_asset(asset, limit, offset)
                self._send(200, {"asset": asset, "items": items, "total": total, "limit": limit, "offset": offset, "status": status})
            except (ValueError, OSError) as exc:
                self._send(200, {"asset": asset, "items": [], "total": 0, "message": str(exc)})
            return
        if parsed.path == "/predict":
            asset = parse_qs(parsed.query).get("asset", [None])[0]
            if not asset or asset not in KNOWN_ASSETS:
                self._send(400, {"error": f"unknown asset {asset!r}"})
                return
            try:
                self._send(200, predict_range_regime(asset, latest_bars(asset)))
            except ValueError as exc:
                self._send(422, {"error": str(exc)})
            return
        # --- React webapp (website/dist/): static assets + SPA fallback. ---
        # API routes above take precedence; everything else belongs to the UI.
        candidate = _webapp_path(parsed.path)
        if candidate is not None:
            suffix = candidate.suffix.lower()
            content_type = _STATIC_CONTENT_TYPES.get(suffix, "application/octet-stream")
            cache = "no-store" if suffix == ".html" else "public, max-age=31536000, immutable"
            self._send_file(candidate, content_type, cache)
            return
        # Extensionless paths (/markets, /login, /register) are React routes.
        if "." not in parsed.path.rsplit("/", 1)[-1]:
            self._serve_webapp_index()
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/refresh":
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            allowed_origins = {f"http://{host}", f"https://{host}"} | set(DEV_BROWSER_ORIGINS)
            if origin and origin not in allowed_origins:
                self._send(403, {"error": "cross-origin refresh requests are not allowed"})
                return
            if not _REFRESH_LOCK.acquire(blocking=False):
                self._send(409, {"error": "a market refresh is already in progress"})
                return
            try:
                assets_updated = refresh_market_data()
                outcomes_scored = 0
                predictions_added = 0
                shadow_error = None
                try:
                    outcomes_scored = label_shadow_outcomes()
                    predictions_added = log_shadow_predictions()
                except Exception:
                    # Full detail goes to the server log only; callers get a
                    # generic note so filesystem/model internals never leak.
                    logger.exception("Shadow scoring/logging failed during /refresh")
                    shadow_error = "shadow logging failed; see server log"
                # News refresh is best-effort: provider outages or missing
                # API tokens must never fail the market refresh.
                news_summary = None
                news_error = None
                try:
                    news_summary = refresh_news_data()
                except Exception:
                    logger.exception("Live news refresh failed during /refresh")
                    news_error = "news refresh failed; see server log"
                market = load_live_market()
                self._send(200, {
                    "assets_updated": assets_updated,
                    "predictions_added": predictions_added,
                    "outcomes_scored": outcomes_scored,
                    "shadow_error": shadow_error,
                    "news": news_summary,
                    "news_error": news_error,
                    "feed": feed_status(market),
                })
            except Exception:
                logger.exception("Market refresh failed")
                self._send(502, {"error": "Market refresh failed. Check the server log and try again."})
            finally:
                _REFRESH_LOCK.release()
            return
        if parsed.path != "/predict":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            asset = payload.get("asset")
            bars = pd.DataFrame(payload.get("bars", []))
            if asset not in KNOWN_ASSETS:
                self._send(400, {"error": f"unknown asset {asset!r}"})
                return
            self._send(200, predict_range_regime(asset, bars))
        except ValueError as exc:
            self._send(422, {"error": str(exc)})
        except (json.JSONDecodeError, KeyError) as exc:
            self._send(400, {"error": f"bad request: {exc}"})

    def log_message(self, *args):  # quieter logs
        pass


class DashboardHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    server = DashboardHTTPServer((host, port), Handler)
    logger.info("AlphaSense volatility API on http://%s:%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
