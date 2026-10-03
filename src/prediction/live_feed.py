"""Shared Upstox V3 market stream, fanned out to local dashboard clients."""

import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("alphasense.feed")

ASSETS_BY_KEY = {
    "NSE_EQ|INE002A01018": "RELIANCE",
    "NSE_EQ|INE467B01029": "TCS",
    "NSE_EQ|INE040A01034": "HDFCBANK",
    "NSE_EQ|INE009A01021": "INFY",
    "NSE_EQ|INE090A01021": "ICICIBANK",
}


class LiveMarketFeed:
    def __init__(self):
        self._lock = threading.Lock()
        self._clients: set[queue.Queue] = set()
        self._streamer = None
        self._status = "disconnected"
        self._restart_timer = None
        self._auth_failed = False
        self._minute_bars: dict[str, dict] = {}
        self._last_tick_status = 0.0

    def subscribe(self) -> queue.Queue:
        client = queue.Queue(maxsize=32)
        with self._lock:
            self._clients.add(client)
            self._start_locked()
            client.put_nowait({"type": "status", "status": self._status})
        return client

    def unsubscribe(self, client: queue.Queue) -> None:
        with self._lock:
            self._clients.discard(client)

    def _start_locked(self) -> None:
        if self._streamer is not None:
            return
        try:
            load_dotenv(Path(__file__).resolve().parents[2] / ".env")
            token = os.getenv("UPSTOX_ACCESS_TOKEN")
            if not token:
                raise RuntimeError("UPSTOX_ACCESS_TOKEN is missing")
            import upstox_client

            configuration = upstox_client.Configuration()
            configuration.access_token = token
            client = upstox_client.ApiClient(configuration)
            streamer = upstox_client.MarketDataStreamerV3(
                client, list(ASSETS_BY_KEY), "full"
            )
            streamer.auto_reconnect(True, interval=5, retry_count=20)
            streamer.on("open", lambda *_: self._on_open())
            streamer.on("close", lambda code=None, reason=None, *_: self._on_close(streamer, code, reason))
            streamer.on("error", self._on_error)
            streamer.on("reconnecting", lambda message, *_: self._set_status(str(message)))
            streamer.on("autoReconnectStopped", lambda message, *_: self._restart_after_exhaustion(streamer, message))
            streamer.on("message", self._on_message)
            self._streamer = streamer
            self._status = "connecting"
            streamer.connect()
        except Exception as exc:
            self._status = f"error: {exc}"
            self._broadcast({"type": "status", "status": self._status})

    def _on_close(self, streamer, code, reason) -> None:
        if self._auth_failed:
            self._set_status("authentication failed (HTTP 401); renew UPSTOX_ACCESS_TOKEN")
            return
        detail = f" ({code}: {reason})" if code is not None or reason else ""
        logger.warning("WebSocket closed%s", detail or " without a close code or reason")
        # Upstox's SDK retries abnormal closes itself, but treats a clean 1000
        # close as final. Surface the reason and explicitly restart that case.
        if code == 1000:
            self._set_status(f"disconnected{detail}; reconnecting")
            self._schedule_restart(streamer)
        else:
            self._set_status(f"reconnecting after disconnect{detail}")

    def _on_error(self, error) -> None:
        detail = str(error)
        if "401 Unauthorized" in detail:
            self._auth_failed = True
            logger.error("Upstox rejected the access token (HTTP 401). Renew UPSTOX_ACCESS_TOKEN.")
            self._set_status("authentication failed (HTTP 401); renew UPSTOX_ACCESS_TOKEN")
            return
        token = os.getenv("UPSTOX_ACCESS_TOKEN")
        if token:
            detail = detail.replace(token, "[redacted]")
        logger.warning("WebSocket error: %s", detail)
        self._set_status(f"error: {detail}")

    def _on_open(self) -> None:
        self._auth_failed = False
        self._set_status("connected")

    def _restart_after_exhaustion(self, streamer, message) -> None:
        self._set_status(f"{message}; restarting feed")
        self._schedule_restart(streamer)

    def _schedule_restart(self, streamer) -> None:
        with self._lock:
            if self._restart_timer and self._restart_timer.is_alive():
                return
            timer = threading.Timer(5, self._restart, args=(streamer,))
            timer.daemon = True
            self._restart_timer = timer
            timer.start()

    def _restart(self, streamer) -> None:
        with self._lock:
            self._restart_timer = None
            if self._streamer is not streamer or not self._clients:
                return
            self._status = "reconnecting"
            self._broadcast({"type": "status", "status": self._status})
        try:
            streamer.connect()
        except Exception as exc:
            self._set_status(f"error: {exc}")
            self._schedule_restart(streamer)

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._broadcast({"type": "status", "status": status})

    def _broadcast(self, message: dict) -> None:
        for client in tuple(self._clients):
            try:
                client.put_nowait(message)
            except queue.Full:
                try:
                    client.get_nowait()
                    client.put_nowait(message)
                except (queue.Empty, queue.Full):
                    pass

    def _on_message(self, message: dict) -> None:
        if not isinstance(message, dict):
            return
        for instrument_key, feed in message.get("feeds", {}).items():
            asset = ASSETS_BY_KEY.get(instrument_key)
            if not asset:
                continue
            full = feed.get("fullFeed", {}) if isinstance(feed, dict) else {}
            market = full.get("marketFF", {})
            market_ohlc = market.get("marketOHLC", {})
            candles = market_ohlc.get("ohlc", [])
            candle = next((item for item in candles if item.get("interval") == "I1"), None)
            try:
                if candle:
                    minute_bar = {
                        "time": int(candle["ts"]) // 1000,
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                    }
                    self._minute_bars[asset] = minute_bar
                else:
                    # OHLC is not present in every Upstox full-feed update.
                    # Build the active 1-minute candle from the LTP ticks so
                    # the chart still moves when only ltpc is sent.
                    ltpc = market.get("ltpc", {})
                    price = float(ltpc["ltp"])
                    raw_time = ltpc.get("ltt")
                    if raw_time:
                        try:
                            tick_time = int(raw_time)
                            if tick_time > 100_000_000_000:
                                tick_time //= 1000
                        except (TypeError, ValueError):
                            tick_time = int(datetime.fromisoformat(str(raw_time)).timestamp())
                    else:
                        tick_time = int(time.time())
                    minute = tick_time - tick_time % 60
                    previous = self._minute_bars.get(asset)
                    if previous and previous["time"] == minute:
                        minute_bar = {
                            **previous,
                            "high": max(previous["high"], price),
                            "low": min(previous["low"], price),
                            "close": price,
                        }
                    else:
                        minute_bar = {
                            "time": minute,
                            "open": price,
                            "high": price,
                            "low": price,
                            "close": price,
                        }
                    self._minute_bars[asset] = minute_bar
            except (KeyError, TypeError, ValueError):
                continue
            event = {"type": "candle", "asset": asset, "bar": minute_bar}
            with self._lock:
                self._broadcast(event)
                now = time.monotonic()
                if now - self._last_tick_status >= 15:
                    self._last_tick_status = now
                    ist = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30)))
                    self._status = f"connected · last tick {ist:%H:%M:%S} IST"
                    self._broadcast({"type": "status", "status": self._status})


live_market_feed = LiveMarketFeed()
