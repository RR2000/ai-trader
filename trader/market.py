"""Candles from Binance's public market data (no API key): one-minute ones for the 2023 model, any interval for the rest."""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

log = logging.getLogger(__name__)

# data-api.binance.vision serves only public market data; api.binance.com is the fallback
BASE_URLS = ("https://data-api.binance.vision", "https://api.binance.com")
MINUTE_MS = 60_000
# the kline intervals Binance serves, in milliseconds (no 1w/1M: their length is not fixed)
INTERVALS = {"1m": MINUTE_MS, "3m": 3 * MINUTE_MS, "5m": 5 * MINUTE_MS, "15m": 15 * MINUTE_MS, "30m": 30 * MINUTE_MS,
             "1h": 60 * MINUTE_MS, "2h": 120 * MINUTE_MS, "4h": 240 * MINUTE_MS, "6h": 360 * MINUTE_MS,
             "8h": 480 * MINUTE_MS, "12h": 720 * MINUTE_MS, "1d": 1440 * MINUTE_MS}
MAX_LIMIT = 1000  # most candles one klines request returns


class MarketDataError(Exception):
    pass


@dataclass(frozen=True)
class Candles:
    open_time_ms: list[int]
    close: list[float]
    volume: list[float]
    interval_ms: int = MINUTE_MS

    @property
    def last_close_time_ms(self) -> int:
        """End of the latest candle (its open time + the interval)."""
        return self.open_time_ms[-1] + self.interval_ms

    def tail(self, n: int) -> "Candles":
        """The latest n candles: each strategy sees exactly the history it asked for."""
        if n >= len(self.open_time_ms):
            return self
        return Candles(self.open_time_ms[-n:], self.close[-n:], self.volume[-n:], self.interval_ms)


def parse_klines(rows: list, now_ms: int, interval_ms: int = MINUTE_MS) -> Candles:
    """Keep the closed candles only, oldest first, and refuse anything that is not a sane price."""
    if not isinstance(rows, list):
        raise MarketDataError("unexpected klines payload")
    opens, closes, volumes = [], [], []
    for row in rows:
        try:
            open_time, close, volume, close_time = int(row[0]), float(row[4]), float(row[5]), int(row[6])
        except (TypeError, ValueError, IndexError) as e:
            raise MarketDataError(f"bad kline row: {e}") from None
        if close_time >= now_ms:
            continue  # still forming
        if not (math.isfinite(close) and close > 0 and math.isfinite(volume) and volume >= 0):
            raise MarketDataError("kline with an invalid price or volume")
        if opens and open_time <= opens[-1]:
            raise MarketDataError("klines out of order")
        opens.append(open_time)
        closes.append(close)
        volumes.append(volume)
    if not opens:
        raise MarketDataError("no closed candles")
    return Candles(opens, closes, volumes, interval_ms)


def _get_json(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": "ai-trader-paper-trading", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read(4 * 1024 * 1024)
    return json.loads(body)


def fetch_candles(symbol: str, limit: int, *, interval: str = "1m", timeout: float = 10.0, attempts: int = 3,
                  backoff: float = 2.0, base_urls: tuple[str, ...] = BASE_URLS, clock=time.time, sleep=time.sleep,
                  get_json=_get_json) -> Candles:
    """The latest `limit` candles of `interval` (closed ones only). Retries, then raises MarketDataError."""
    interval_ms = INTERVALS[interval]
    query = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    last_error: Exception | None = None
    for attempt in range(attempts):
        base = base_urls[attempt % len(base_urls)]
        try:
            rows = get_json(f"{base}/api/v3/klines?{query}", timeout)
            return parse_klines(rows, int(clock() * 1000), interval_ms)
        except (OSError, ValueError, MarketDataError) as e:  # URLError, timeouts and bad JSON included
            last_error = e
            log.warning("klines from %s failed (attempt %d/%d): %s", base, attempt + 1, attempts, e)
            if attempt + 1 < attempts:
                sleep(backoff * (attempt + 1))
    raise MarketDataError(f"Binance market data unavailable: {last_error}")
