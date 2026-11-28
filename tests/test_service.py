"""The worker never crashes, one failing simulation does not stop the others, the pages answer."""

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from conftest import simulations
from trader.market import INTERVALS, Candles, MarketDataError
from trader.service import Cache, Worker, build_state, build_summary, make_handler, next_tick

SIMS = simulations("""
[[simulation]]
id = "dqn-2023"
name = "DQN 2023"
strategy = "dqn"
[[simulation]]
id = "hold"
name = "Hold"
strategy = "buy-and-hold"
[[simulation]]
id = "sma"
name = "SMA"
strategy = "sma-cross"
interval = "15m"
[simulation.params]
fast = 3
slow = 8
""")
NOW = datetime(2026, 11, 26, 12, 0, 3, tzinfo=timezone.utc)


def fake_candles(n, interval="1m", now=NOW):
    rng = np.random.default_rng(3)
    ms = INTERVALS[interval]
    end_ms = int(now.timestamp() * 1000) // ms * ms
    opens = [end_ms - (n - i) * ms for i in range(n)]
    close = list(60_000 * np.exp(np.cumsum(rng.normal(0, 0.001, n))))
    return Candles(opens, close, list(rng.gamma(2.0, 4.0, n)), ms)


class Market:
    """Binance as the worker sees it: the closed candles of a request (limit - 1, the last one is still forming)."""

    def __init__(self, down=(), now=NOW):
        self.calls, self.down, self.now = [], set(down), now

    def __call__(self, symbol, limit, interval="1m"):
        self.calls.append((symbol, interval, limit))
        if interval in self.down:
            raise MarketDataError("down")
        return fake_candles(limit - 1, interval, self.now)


class BrokenStore:
    def init_schema(self):
        raise OSError("connection refused")

    def snapshot(self, sim_id, since):
        raise OSError("connection refused")

    def summary(self, ids):
        raise OSError("connection refused")


def worker(store, market=None, sims=SIMS):
    market = market or Market()
    return Worker(store, sims, fetch=market, clock=lambda: market.now.timestamp())


def test_binance_down_is_not_a_crash():
    w = worker(BrokenStore(), Market(down={"1m", "15m"}))
    assert w.tick() == {}
    assert {st.last_error for st in w.status.values()} == {"Binance market data unavailable"}


def test_database_down_is_not_a_crash():
    w = worker(BrokenStore())
    assert w.tick() == {}
    assert w.status["dqn-2023"].last_candle is None  # nothing was recorded
    assert w.status["sma"].last_error == "database unavailable"
    state = build_state(BrokenStore(), w, "sma", "7d")
    assert state["status"]["stale"] is True and state["status"]["error"] == "database unavailable"
    summary = build_summary(BrokenStore(), w)
    assert summary["error"] == "database unavailable" and len(summary["simulations"]) == 3


def test_one_request_per_pair_and_interval(store):
    market = Market()
    w = worker(store, market)
    steps = w.tick()
    assert set(steps) == {"dqn-2023", "hold", "sma"}
    # the 1m candles once for two simulations, 500 of them as the first release asked; the 15m ones once
    assert sorted(market.calls) == [("BTCUSDT", "15m", 9), ("BTCUSDT", "1m", 500)]
    assert steps["hold"].trade.side == "buy"
    assert w.tick() == {}  # same minute: no new candle is due, nothing is fetched
    assert len(market.calls) == 2
    market.now = NOW + timedelta(minutes=1)
    w.tick()
    assert [c[1] for c in market.calls[2:]] == ["1m"]  # the 15m candle is not due yet


def test_a_failing_simulation_does_not_stop_the_others(store, monkeypatch):
    w = worker(store)

    def broken(candles, position):
        raise ValueError("bug")

    monkeypatch.setattr(SIMS[2].strategy, "decide", broken)
    steps = w.tick()
    assert set(steps) == {"dqn-2023", "hold"}
    assert w.status["sma"].last_error == "the strategy failed on the latest candles"
    assert w.status["dqn-2023"].last_error is None
    monkeypatch.undo()
    assert set(w.tick()) == {"sma"}  # retried on the next pass, the others already have this candle
    assert w.status["sma"].last_error is None


def test_a_changed_definition_stops_only_that_simulation(store):
    w = worker(store)
    w.tick()
    changed = simulations('[[simulation]]\nid = "hold"\nname = "Hold"\nstrategy = "buy-and-hold"\nstart_cash = 5\n')
    w2 = worker(store, Market(now=NOW + timedelta(minutes=1)), changed + [SIMS[0]])
    steps = w2.tick()
    assert set(steps) == {"dqn-2023"}
    assert w2.status["hold"].disabled and "stopped" in w2.status["hold"].last_error


def test_tick_records_a_decision_and_the_page_state(store):
    w = worker(store)
    step = w.tick()["dqn-2023"]
    assert step.price == fake_candles(499).close[-1]
    down = build_state(BrokenStore(), w, "dqn-2023", "7d")  # database gone: the page still knows the last decision
    assert down["status"]["stale"] and down["status"]["last_candle"] == step.ts.isoformat().replace("+00:00", "Z")
    state = build_state(store, w, "dqn-2023", "all", now=step.ts + timedelta(seconds=30))
    assert state["status"]["stale"] is False
    assert state["portfolio"]["value"] == pytest.approx(step.value)
    assert state["portfolio"]["last_q"] is not None and state["portfolio"]["last_reason"].startswith("Q-values")
    assert state["series"][0][1] == 1000.0 and len(state["series"]) == 2
    assert state["simulation"]["strategy"] == "dqn" and state["simulation"]["params"][0]["name"] == "model"
    assert build_state(store, w, "dqn-2023", "1d", now=step.ts + timedelta(minutes=10))["status"]["stale"] is True
    sma_ts = w.status["sma"].last_candle
    # a 15-minute simulation is not stale 10 minutes after its candle, it is after 17
    assert build_state(store, w, "sma", "1d", now=sma_ts + timedelta(minutes=10))["status"]["stale"] is False
    assert build_state(store, w, "sma", "1d", now=sma_ts + timedelta(minutes=18))["status"]["stale"] is True
    summary = build_summary(store, w, now=step.ts + timedelta(seconds=30))
    rows = {s["id"]: s for s in summary["simulations"]}
    assert [s["id"] for s in summary["simulations"]] == ["dqn-2023", "hold", "sma"]  # the order of the file
    assert rows["hold"]["portfolio"]["trades"] == 1 and rows["hold"]["portfolio"]["position"] == "BTC"
    assert rows["hold"]["portfolio"]["value"] == pytest.approx(rows["hold"]["portfolio"]["benchmark"])
    # one candle: the risk statistics wait for more history, and say how much there is
    assert rows["hold"]["risk"]["strategy"] == {**rows["hold"]["risk"]["strategy"], "enough": False, "returns": 1,
                                                 "volatility": None, "sharpe": None, "max_drawdown": None}
    assert state["risk"]["buy_and_hold"]["enough"] is False and state["risk_rules"]["min_days"] == 7


@pytest.fixture
def server(store):
    w = worker(store)
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
    assert status == 200 and b"<h1>AI Trader</h1>" in body and b"not financial advice" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    for path in ("/app.css", "/common.js", "/home.js", "/sim.js", "/favicon.svg"):
        assert get(server + path)[0] == 200, path
    status, _, body = get(server + "/s/sma")
    assert status == 200 and b"/sim.js" in body and b"not financial advice" in body
    assert get(server + "/s/sma/")[0] == 200
    status, _, body = get(server + "/s/nope")
    assert status == 404 and b"not financial advice" in body
    status, headers, body = get(server + "/api/simulations")
    summary = json.loads(body)
    assert status == 200 and headers["Cache-Control"] == "no-store"
    assert [s["id"] for s in summary["simulations"]] == ["dqn-2023", "hold", "sma"]
    state = json.loads(get(server + "/api/simulations/sma?range=1d")[2])
    assert state["portfolio"]["start_cash"] == 1000.0 and state["simulation"]["name"] == "SMA"
    legacy = json.loads(get(server + "/api/state?range=1d")[2])  # the first page's address still works
    assert legacy["simulation"]["id"] == "dqn-2023"
    assert get(server + "/api/simulations/sma?range=bogus")[0] == 400
    assert get(server + "/api/simulations/nope")[0] == 404
    assert get(server + "/api/nope")[0] == 404
    assert get(server + "/../etc/passwd")[0] == 404
    assert json.loads(get(server + "/healthz")[2])["ok"] is True


def test_a_slow_pass_does_not_skip_a_candle():
    base = 1_795_000_000 // 60 * 60  # a minute boundary
    assert next_tick(base + 3) == base + 63  # a pass on time: the next one a minute later
    assert next_tick(base - 4) == base + 3  # started before the boundary, still running after it: again at once
    assert next_tick(base + 59) == base + 63
