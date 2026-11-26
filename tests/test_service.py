"""The worker never crashes, the page says when data is stale, the web server answers."""

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from conftest import MODEL
from trader.market import Candles, MarketDataError
from trader.model import QNetwork
from trader.service import Cache, Worker, build_state, make_handler

NET = QNetwork(str(MODEL))


def fake_candles(n=500, end_ms=None):
    rng = np.random.default_rng(3)
    end_ms = end_ms or int(datetime.now(timezone.utc).timestamp() // 60 * 60_000)
    opens = [end_ms - (n - i) * 60_000 for i in range(n)]
    close = list(60_000 * np.exp(np.cumsum(rng.normal(0, 0.001, n))))
    return Candles(opens, close, list(rng.gamma(2.0, 4.0, n)))


class BrokenStore:
    def init_schema(self):
        raise OSError("connection refused")

    def snapshot(self, since):
        raise OSError("connection refused")


def test_binance_down_is_not_a_crash():
    def fetch(symbol, limit):
        raise MarketDataError("down")

    w = Worker(BrokenStore(), NET, "BTCUSDT", "m.h5", fetch=fetch)
    assert w.tick() is None
    assert w.last_error == "Binance market data unavailable"


def test_database_down_is_not_a_crash():
    w = Worker(BrokenStore(), NET, "BTCUSDT", "m.h5", fetch=lambda s, n: fake_candles())
    assert w.tick() is None
    assert w.last_candle is None  # nothing was recorded
    assert w.last_error == "database unavailable"
    state = build_state(BrokenStore(), w, "7d")
    assert state["status"]["stale"] is True and state["status"]["error"] == "database unavailable"


def test_tick_records_a_decision_and_the_page_state(store):
    candles = fake_candles()
    w = Worker(store, NET, "BTCUSDT", "m.h5", fetch=lambda s, n: candles)
    step = w.tick()
    assert step is not None and w.last_error is None
    assert step.price == candles.close[-1]
    assert w.tick() is None  # same candle: nothing new
    assert w.last_candle == step.ts
    down = build_state(BrokenStore(), w, "7d")  # database gone: the page still knows the last decision
    assert down["status"]["stale"] and down["status"]["last_candle"] == step.ts.isoformat().replace("+00:00", "Z")
    now = step.ts + timedelta(seconds=30)
    state = build_state(store, w, "all", now=now)
    assert state["status"]["stale"] is False
    assert state["portfolio"]["value"] == pytest.approx(step.value)
    assert state["series"][0][1] == 1000.0 and len(state["series"]) == 2
    later = build_state(store, w, "1d", now=step.ts + timedelta(minutes=10))
    assert later["status"]["stale"] is True  # no decision for 10 minutes


@pytest.fixture
def server(store):
    w = Worker(store, NET, "BTCUSDT", "m.h5", fetch=lambda s, n: fake_candles())
    w.tick()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, w, Cache(ttl=0)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_web_server(server):
    status, headers, body = get(server + "/")
    assert status == 200 and b"ai-trader" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    for path in ("/app.js", "/app.css", "/favicon.svg"):
        assert get(server + path)[0] == 200
    status, headers, body = get(server + "/api/state?range=1d")
    state = json.loads(body)
    assert status == 200 and headers["Cache-Control"] == "no-store"
    assert state["portfolio"]["start_cash"] == 1000.0 and state["status"]["stale"] is False
    assert get(server + "/api/state?range=bogus")[0] == 400
    assert get(server + "/../etc/passwd")[0] == 404
    assert json.loads(get(server + "/healthz")[2])["ok"] is True
