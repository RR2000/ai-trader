"""The live paper-trading service: a worker that runs every simulation once a minute and a small web server.

Every minute, a few seconds after a candle closes, the worker reads the latest closed candles from Binance's
public market data once per pair and interval, hands them to every simulation that uses them and records each
decision in PostgreSQL. Binance or the database being down never stops the process, and a simulation that
fails never stops the others: that candle is skipped and the page says the data is stale.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config as config_mod
from . import metrics
from .config import Simulation
from .market import MAX_LIMIT, fetch_candles
from .store import LEGACY_SIMULATION, ConfigChanged, Store, StrategyFailed, utc
from .strategies import ACTIONS, ConfigError

log = logging.getLogger("trader")

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).parent / "static"
STALE_GRACE = timedelta(minutes=2)  # a simulation is stale when its next candle is this late (3 min for 1m candles)
RANGES = {"1d": timedelta(days=1), "7d": timedelta(days=7), "30d": timedelta(days=30), "all": None}
CACHE_SECONDS = 15
RISK_SECONDS = 300  # the risk statistics read the whole history: once every five minutes is plenty


@dataclass
class SimStatus:
    sim: Simulation
    last_error: str | None = None  # short, public text: shown on the page
    last_ok_at: datetime | None = None
    last_candle: datetime | None = None  # latest candle recorded, for the page when the database is down
    done_ms: int = 0  # end of the latest candle handled (recorded, or found already recorded)
    disabled: bool = False  # its definition no longer matches its stored portfolio


class Worker:
    def __init__(self, store: Store, sims: list[Simulation], *, fetch=fetch_candles, clock=time.time):
        self.store, self.sims, self.fetch, self.clock = store, sims, fetch, clock
        self.status = {s.id: SimStatus(s) for s in sims}
        self.groups: dict[tuple[str, str], list[SimStatus]] = {}
        for s in sims:
            self.groups.setdefault((s.symbol, s.interval), []).append(self.status[s.id])
        self.schema_ready = False
        self.lock = threading.Lock()
        self.last_ok_at: datetime | None = None

    def _fail(self, st: SimStatus, message: str):
        with self.lock:
            st.last_error = message

    def tick(self) -> dict:
        """One pass: every pair and interval with a new closed candle due. Returns the steps recorded, by id."""
        steps = {}
        now_ms = int(self.clock() * 1000)
        for (symbol, interval), members in self.groups.items():
            live = [st for st in members if not st.disabled]
            if not live or all(now_ms < st.done_ms + st.sim.interval_ms for st in live):
                continue  # no new candle can have closed yet
            limit = min(MAX_LIMIT, max(st.sim.strategy.lookback for st in live) + 1)  # +1: the candle still forming
            try:
                data = self.fetch(symbol, limit, interval=interval)
            except Exception as e:  # noqa: BLE001 - nothing must stop the loop
                log.warning("market data %s %s: %s", symbol, interval, e)
                for st in live:
                    self._fail(st, "Binance market data unavailable")
                continue
            for st in live:
                step = self._step(st, data)
                if step is not None:
                    steps[st.sim.id] = step
        return steps

    def _step(self, st: SimStatus, data):
        sim = st.sim
        ts, price = utc(data.last_close_time_ms), data.close[-1]
        candles = data.tail(sim.strategy.lookback)
        try:
            if not self.schema_ready:
                if self.store.init_schema():
                    log.info("database migrated: the history of the single simulation is now %s", LEGACY_SIMULATION)
                self.schema_ready = True
            step = self.store.record(sim, ts, price, lambda pos: sim.strategy.decide(candles, pos))
        except ConfigChanged:
            log.error("%s: simulations.toml no longer matches how it started (strategy, parameters, pair, interval,"
                      " cash or fee); it is stopped: give the new definition a new id", sim.id)
            with self.lock:
                st.disabled, st.last_error = True, "definition changed since it started: stopped"
            return None
        except StrategyFailed as e:
            log.error("%s: strategy failed: %r", sim.id, e.__cause__)
            self._fail(st, "the strategy failed on the latest candles")
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("%s database: %s", sim.id, type(e).__name__)
            self._fail(st, "database unavailable")
            return None
        with self.lock:
            st.last_error, st.last_ok_at = None, datetime.now(timezone.utc)
            st.last_candle, st.done_ms = ts, data.last_close_time_ms
            self.last_ok_at = st.last_ok_at
        if step is None:
            log.info("%s: candle %s already processed", sim.id, ts.isoformat())
        else:
            log.info("%s %s price=%.2f action=%s trade=%s value=%.2f buy&hold=%.2f (%s)", sim.id, ts.isoformat(),
                     price, ACTIONS[step.action], step.trade.side if step.trade else "-", step.value, step.benchmark,
                     step.reason)
        return step

    def run(self, stop: threading.Event, delay: float = 3.0):
        """Tick once now, then a few seconds after every minute boundary, until stop is set."""
        while not stop.is_set():
            started = time.time()
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - a bug must not kill the worker thread
                log.exception("tick failed")
            # the next minute after the one this pass started in: a pass that runs past the boundary is followed
            # at once by the next one, so no candle is skipped
            stop.wait(max(0.0, next_tick(started, delay) - time.time()))


def next_tick(started: float, delay: float = 3.0) -> float:
    """When the pass after one started at `started` runs: `delay` seconds after the next minute boundary."""
    return (started - delay) // 60 * 60 + 60 + delay


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def describe(sim: Simulation) -> dict:
    s = sim.strategy
    return {"id": sim.id, "name": sim.name, "description": sim.description, "strategy": s.kind,
            "strategy_title": s.title, "explain": s.explain(sim.quote),
            "params": [{"name": k, "value": v, "help": s.PARAMS[k].help} for k, v in s.params.items()],
            "symbol": sim.symbol, "base": sim.base, "quote": sim.quote, "interval": sim.interval,
            "start_cash": sim.start_cash, "fee": sim.fee}


def _status(worker: Worker, st: SimStatus, portfolio: dict | None, now: datetime) -> dict:
    with worker.lock:
        error = st.last_error
    last_candle = portfolio["last_candle"] if portfolio else None
    late = timedelta(milliseconds=st.sim.interval_ms) + STALE_GRACE
    return {"stale": last_candle is None or now - last_candle > late, "error": error, "last_candle": _iso(last_candle),
            "updated_at": _iso(portfolio["updated_at"]) if portfolio else None}


def _down(worker: Worker, st: SimStatus) -> dict:
    with worker.lock:
        known = st.last_candle
    return {"stale": True, "error": "database unavailable", "last_candle": _iso(known), "updated_at": None}


def _position(sim: Simulation, p: dict) -> str:
    if p["crypto"] > 0 and p["cash"] > 0.005:
        return f"{sim.base} + {sim.quote}"
    return sim.base if p["crypto"] > 0 else sim.quote


def build_risk(store: Store, sim: Simulation) -> dict | None:
    """Volatility, maximum drawdown and Sharpe ratio of the simulation and of its buy and hold, over the whole
    history (trader/metrics.py). None before the first decision or when the database cannot answer."""
    try:
        r = store.risk(sim.id)
    except Exception as e:  # noqa: BLE001
        log.warning("database read (risk): %s", type(e).__name__)
        return None
    if r is None:
        return None
    span = (_ms(r["last_ts"]) - _ms(r["started_at"]) + sim.interval_ms) if r["last_ts"] else 0
    out = {}
    for key, k in (("strategy", "v"), ("buy_and_hold", "b")):
        m = metrics.finish(r["n"], r[f"mean_{k}"], r[f"std_{k}"], r[f"low_{k}"], r[f"low_{k}_at"], span,
                           sim.interval_ms)
        m["max_drawdown_at"] = _iso(m["max_drawdown_at"])
        out[key] = m
    return out


def _risk(store: Store, sim: Simulation, risk_cache: Cache | None):
    if risk_cache is None:
        return build_risk(store, sim)
    return risk_cache.get(sim.id, lambda: build_risk(store, sim))


def build_state(store: Store, worker: Worker, sim_id: str, range_key: str, now: datetime | None = None,
                risk_cache: Cache | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    st = worker.status[sim_id]
    sim = st.sim
    state = {"now": _iso(now), "simulation": describe(sim), "symbol": sim.symbol, "range": range_key,
             "actions": list(ACTIONS), "portfolio": None, "series": [], "trades": [], "stats": None, "risk": None,
             "risk_rules": RISK_RULES}
    since = RANGES[range_key]
    try:
        snap = store.snapshot(sim_id, now - since if since else None)
    except Exception as e:  # noqa: BLE001
        log.warning("database read: %s", type(e).__name__)
        state["status"] = _down(worker, st)
        return state
    p, last = snap["portfolio"], snap.get("last")
    state["status"] = _status(worker, st, p, now)
    if p is None:
        return state
    price = last["price"] if last else p["start_price"]
    state["portfolio"] = {
        "started_at": _iso(p["started_at"]), "start_price": p["start_price"], "start_cash": p["start_cash"],
        "fee": p["fee"], "cash": p["cash"], "crypto": p["crypto"], "price": price,
        "value": p["cash"] + p["crypto"] * price, "benchmark": p["bench_units"] * price,
        "position": _position(sim, p),
        "last_action": ACTIONS[last["action"]] if last else None,
        "last_reason": p["last_reason"],
        "last_q": [last["q_buy"], last["q_sell"], last["q_hold"]] if last and last["q_buy"] is not None else None,
    }
    series = [[_ms(r["ts"]), round(r["value"], 4), round(r["benchmark"], 4), r["price"]] for r in snap["series"]]
    if since is None or now - since <= p["started_at"]:
        # the whole history is in view: both lines start from the initial cash, a candle before the first decision
        series.insert(0, [_ms(p["started_at"]) - sim.interval_ms, p["start_cash"], p["start_cash"], p["start_price"]])
    state["series"] = series
    state["trades"] = [{"ts": _iso(t["ts"]), "side": t["side"], "price": t["price"], "quantity": t["quantity"],
                        "fee_usdt": t["fee_usdt"], "value_after": t["value_after"]} for t in snap["trades"]]
    s = snap["stats"]
    state["stats"] = {"trades": s["trades"], "fees": s["fees"], "decisions": s["decisions"],
                      "last_24h": {"buy": s["buy_24h"], "sell": s["sell_24h"], "hold": s["hold_24h"]}}
    state["risk"] = _risk(store, sim, risk_cache)
    return state


def build_summary(store: Store, worker: Worker, now: datetime | None = None, risk_cache: Cache | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    try:
        rows = store.summary([s.id for s in worker.sims])
        error = None
    except Exception as e:  # noqa: BLE001
        log.warning("database read: %s", type(e).__name__)
        rows, error = {}, "database unavailable"
    out = []
    for sim in worker.sims:
        st = worker.status[sim.id]
        d = describe(sim)
        item = {k: d[k] for k in ("id", "name", "description", "strategy", "strategy_title", "symbol", "base", "quote",
                                  "interval", "start_cash")}
        r = rows.get(sim.id)
        item["status"] = _down(worker, st) if error else _status(worker, st, r, now)
        item["portfolio"] = None
        if r is not None:
            price = r["last_price"] if r["last_price"] is not None else r["start_price"]
            first = [_ms(r["started_at"]) - sim.interval_ms, r["start_cash"], r["start_cash"]]
            item["portfolio"] = {
                "started_at": _iso(r["started_at"]), "start_cash": r["start_cash"],
                "value": r["cash"] + r["crypto"] * price, "benchmark": r["bench_units"] * price, "price": price,
                "position": _position(sim, r), "trades": r["trades"],
                "spark": [first] + [[_ms(x["ts"]), round(x["value"], 4), round(x["benchmark"], 4)] for x in r["spark"]],
            }
            item["risk"] = _risk(store, sim, risk_cache)
        out.append(item)
    return {"now": _iso(now), "error": error, "simulations": out, "risk_rules": RISK_RULES}


RISK_RULES = {"min_days": metrics.MIN_DAYS, "min_candles": metrics.MIN_RETURNS, "days_per_year": 365,
              "risk_free_rate": 0}


class Cache:
    def __init__(self, ttl: float = CACHE_SECONDS):
        self.ttl, self.lock, self.items = ttl, threading.Lock(), {}

    def get(self, key, make):
        with self.lock:  # one database read per page and range every ttl seconds, whatever the traffic
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
SIM_PATH = re.compile(r"^/s/([a-z0-9-]{1,40})/?$")
API_SIM_PATH = re.compile(r"^/api/simulations/([a-z0-9-]{1,40})$")


def make_handler(store: Store, worker: Worker, cache: Cache, risk_cache: Cache | None = None):
    risk_cache = risk_cache or Cache(RISK_SECONDS)
    files = {"/": "index.html", "/index.html": "index.html", "/app.css": "app.css", "/common.js": "common.js",
             "/home.js": "home.js", "/sim.js": "sim.js", "/favicon.svg": "favicon.svg"}

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

        def _file(self, name: str, code: int = 200):
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._send(code, (STATIC / name).read_bytes(), ctype, "public, max-age=300" if code == 200 else "no-store")

        def _sim_state(self, sim_id: str, query: str):
            key = parse_qs(query).get("range", ["7d"])[0]
            if key not in RANGES:
                return self._json(400, {"error": "range must be one of " + ", ".join(RANGES)})
            return self._json(200, cache.get((sim_id, key), lambda: build_state(store, worker, sim_id, key, risk_cache=risk_cache)))

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/simulations":
                return self._json(200, cache.get("summary", lambda: build_summary(store, worker, risk_cache=risk_cache)))
            m = API_SIM_PATH.match(url.path)
            if m:
                if m.group(1) not in worker.status:
                    return self._json(404, {"error": "no such simulation"})
                return self._sim_state(m.group(1), url.query)
            if url.path == "/api/state" and LEGACY_SIMULATION in worker.status:  # the first page's API
                return self._sim_state(LEGACY_SIMULATION, url.query)
            if url.path == "/healthz":
                with worker.lock:
                    ok = worker.last_ok_at
                return self._json(200, {"ok": True, "last_ok_at": _iso(ok)})
            m = SIM_PATH.match(url.path)
            if m:
                return self._file("sim.html" if m.group(1) in worker.status else "404.html",
                                  200 if m.group(1) in worker.status else 404)
            name = files.get(url.path)
            if name is None:
                if url.path.startswith("/api/"):
                    return self._json(404, {"error": "not found"})
                return self._file("404.html", 404)
            self._file(name)

    return Handler


def store_from_env() -> Store:
    if os.environ.get("DATABASE_URL"):
        return Store(os.environ["DATABASE_URL"])
    return Store(host=os.environ.get("DB_HOST", "localhost"), port=int(os.environ.get("DB_PORT", "5432")),
                 dbname=os.environ.get("DB_NAME", "trader"), user=os.environ.get("DB_USER", "trader"),
                 password=os.environ.get("DB_PASSWORD", ""))


def load_simulations() -> list[Simulation]:
    path = os.environ.get("SIMULATIONS", str(ROOT / "simulations.toml"))
    model_dir = os.environ.get("MODEL_DIR", os.pathsep.join(str(ROOT / "src" / "model" / d) for d in ("completed", "mlp")))
    return config_mod.load(path, model_dir=model_dir, networks={})


def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        sims = load_simulations()
    except ConfigError as e:
        log.error("%s", e)
        sys.exit(2)
    store = store_from_env()
    worker = Worker(store, sims)
    stop = threading.Event()
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), make_handler(store, worker, Cache()))
    server.daemon_threads = True

    def shutdown(*_):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=worker.run, args=(stop,), name="worker", daemon=True).start()
    log.info("listening on :%d, %d simulations: %s", server.server_address[1], len(sims),
             ", ".join(f"{s.id} ({s.strategy.kind}, {s.symbol} {s.interval})" for s in sims))
    server.serve_forever()
    server.server_close()
