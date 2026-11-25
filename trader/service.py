"""The live paper-trading service: a worker that decides once a minute and a small web server.

Every minute, a few seconds after the candle closes, the worker reads the latest closed one-minute
candles from Binance's public market data, builds the observation the model was trained on, asks
the network for buy, sell or hold and records the result in PostgreSQL. Binance or the database
being down never stops the process: the minute is skipped and the page says the data is stale.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import features
from .market import fetch_candles
from .model import ACTIONS, QNetwork
from .store import Store, utc

log = logging.getLogger("trader")

STATIC = Path(__file__).parent / "static"
STALE_AFTER = timedelta(minutes=3)
RANGES = {"1d": timedelta(days=1), "7d": timedelta(days=7), "30d": timedelta(days=30), "all": None}
CACHE_SECONDS = 15


class Worker:
    def __init__(self, store: Store, net: QNetwork, symbol: str, model_name: str, *, candles: int = 500,
                 fetch=fetch_candles):
        self.store, self.net, self.symbol, self.model_name = store, net, symbol, model_name
        self.candles, self.fetch = candles, fetch
        self.schema_ready = False
        self.lock = threading.Lock()
        self.last_error: str | None = None  # short, public text: shown on the page
        self.last_ok_at: datetime | None = None

    def tick(self):
        try:
            data = self.fetch(self.symbol, self.candles)
        except Exception as e:  # noqa: BLE001 - nothing must stop the loop
            log.warning("market data: %s", e)
            self._fail("Binance market data unavailable")
            return None
        try:
            obs = features.observation(data.close, data.volume)
            action, q = self.net.decide(obs)
        except Exception:  # noqa: BLE001
            log.exception("observation or model failed")
            self._fail("could not build the observation")
            return None
        try:
            if not self.schema_ready:
                self.store.init_schema()
                self.schema_ready = True
            step = self.store.record(utc(data.last_close_time_ms), data.close[-1], action, tuple(q),
                                     symbol=self.symbol, model=self.model_name)
        except Exception as e:  # noqa: BLE001
            log.warning("database: %s", type(e).__name__)
            self._fail("database unavailable")
            return None
        with self.lock:
            self.last_error, self.last_ok_at = None, datetime.now(timezone.utc)
        if step is None:
            log.info("candle %s already processed", utc(data.last_close_time_ms).isoformat())
        else:
            log.info("%s price=%.2f q=%s action=%s trade=%s value=%.2f buy&hold=%.2f", step.ts.isoformat(), step.price,
                     [round(float(v), 4) for v in q], ACTIONS[action], step.trade.side if step.trade else "-",
                     step.value, step.benchmark)
        return step

    def _fail(self, message: str):
        with self.lock:
            self.last_error = message

    def run(self, stop: threading.Event, delay: float = 3.0):
        """Tick once now, then a few seconds after every minute boundary, until stop is set."""
        while not stop.is_set():
            self.tick()
            now = time.time()
            stop.wait(60 - now % 60 + delay)


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def build_state(store: Store, worker: Worker, range_key: str, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    with worker.lock:
        error = worker.last_error
    state = {"now": _iso(now), "symbol": worker.symbol, "range": range_key, "actions": list(ACTIONS),
             "model": worker.model_name, "portfolio": None, "series": [], "trades": [], "stats": None}
    since = RANGES[range_key]
    try:
        snap = store.snapshot(now - since if since else None)
    except Exception as e:  # noqa: BLE001
        log.warning("database read: %s", type(e).__name__)
        state["status"] = {"stale": True, "error": "database unavailable", "last_candle": None, "updated_at": None}
        return state
    p, last = snap["portfolio"], snap.get("last")
    last_candle = p["last_candle"] if p else None
    state["status"] = {
        "stale": last_candle is None or now - last_candle > STALE_AFTER,
        "error": error,
        "last_candle": _iso(last_candle),
        "updated_at": _iso(p["updated_at"]) if p else None,
    }
    if p is None:
        return state
    price = last["price"] if last else p["start_price"]
    state["portfolio"] = {
        "started_at": _iso(p["started_at"]), "start_price": p["start_price"], "start_cash": p["start_cash"],
        "fee": p["fee"], "cash": p["cash"], "crypto": p["crypto"], "price": price,
        "value": p["cash"] + p["crypto"] * price, "benchmark": p["bench_units"] * price,
        "position": "BTC" if p["crypto"] > 0 else "USDT",
        "last_action": ACTIONS[last["action"]] if last else None,
        "last_q": [last["q_buy"], last["q_sell"], last["q_hold"]] if last else None,
    }
    series = [[_ms(r["ts"]), round(r["value"], 4), round(r["benchmark"], 4), r["price"]] for r in snap["series"]]
    if since is None or now - since <= p["started_at"]:
        # the whole history is in view: both lines start from the initial cash, a minute before the first decision
        series.insert(0, [_ms(p["started_at"]) - 60_000, p["start_cash"], p["start_cash"], p["start_price"]])
    state["series"] = series
    state["trades"] = [{"ts": _iso(t["ts"]), "side": t["side"], "price": t["price"], "quantity": t["quantity"],
                        "fee_usdt": t["fee_usdt"], "value_after": t["value_after"]} for t in snap["trades"]]
    s = snap["stats"]
    state["stats"] = {"trades": s["trades"], "fees": s["fees"], "decisions": s["decisions"],
                      "last_24h": {"buy": s["buy_24h"], "sell": s["sell_24h"], "hold": s["hold_24h"]}}
    return state


class Cache:
    def __init__(self, ttl: float = CACHE_SECONDS):
        self.ttl, self.lock, self.items = ttl, threading.Lock(), {}

    def get(self, key, make):
        with self.lock:  # one database read per range every ttl seconds, whatever the traffic
            hit = self.items.get(key)
            if hit and time.monotonic() - hit[0] < self.ttl:
                return hit[1]
            value = make()
            self.items[key] = (time.monotonic(), value)
            return value


SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self';"
                               " connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


def make_handler(store: Store, worker: Worker, cache: Cache):
    files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/app.css": "app.css",
             "/favicon.svg": "favicon.svg"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "trader"
        sys_version = ""
        timeout = 20

        def log_message(self, fmt, *args):  # access log at debug level only
            log.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, code: int, body: bytes, ctype: str, cache_control: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            for k, v in SECURITY_HEADERS.items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, payload):
            self._send(code, json.dumps(payload, separators=(",", ":")).encode(), "application/json", "no-store")

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/state":
                key = parse_qs(url.query).get("range", ["7d"])[0]
                if key not in RANGES:
                    return self._json(400, {"error": "range must be one of " + ", ".join(RANGES)})
                return self._json(200, cache.get(key, lambda: build_state(store, worker, key)))
            if url.path == "/healthz":
                with worker.lock:
                    ok = worker.last_ok_at
                return self._json(200, {"ok": True, "last_ok_at": _iso(ok)})
            name = files.get(url.path)
            if name is None:
                return self._json(404, {"error": "not found"})
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._send(200, (STATIC / name).read_bytes(), ctype, "public, max-age=300")

    return Handler


def store_from_env() -> Store:
    if os.environ.get("DATABASE_URL"):
        return Store(os.environ["DATABASE_URL"])
    return Store(host=os.environ.get("DB_HOST", "localhost"), port=int(os.environ.get("DB_PORT", "5432")),
                 dbname=os.environ.get("DB_NAME", "trader"), user=os.environ.get("DB_USER", "trader"),
                 password=os.environ.get("DB_PASSWORD", ""))


def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    model_path = os.environ.get("MODEL_PATH", str(Path(__file__).parent.parent / "src/model/completed/trading_model_episode_30.h5"))
    net = QNetwork(model_path)
    store = store_from_env()
    worker = Worker(store, net, os.environ.get("SYMBOL", "BTCUSDT"), Path(model_path).name)
    stop = threading.Event()
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), make_handler(store, worker, Cache()))
    server.daemon_threads = True

    def shutdown(*_):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=worker.run, args=(stop,), name="worker", daemon=True).start()
    log.info("listening on :%d, symbol %s, model %s", server.server_address[1], worker.symbol, worker.model_name)
    server.serve_forever()
    server.server_close()
