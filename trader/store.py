"""Persistence on PostgreSQL: per simulation, the portfolio, one row per decision (the value series) and the trades.

The state lives in the database, so a restart or a new deploy continues the same portfolios.
Every decision is one transaction: the equity row, the trade (if any) and the new balances
are written together or not at all, and a candle already processed is never applied twice.

Schema history. Until November 2026 there was one simulation, the 2023 DQN, in the tables portfolio, equity
and trades. init_schema() creates the sim_* tables and, once (schema_migrations), copies those three tables
into them as simulation dqn-2023, in one transaction: all of it or nothing. The old tables are left untouched,
as a backup and so that the previous image still runs if a deploy is rolled back.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .portfolio import Portfolio, Trade, apply, buy_and_hold_units
from .strategies import Decision, Position

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name       text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sim_portfolio (
    simulation  text PRIMARY KEY,
    config      jsonb NOT NULL,
    fee         double precision NOT NULL,
    started_at  timestamptz NOT NULL,
    start_price double precision NOT NULL,
    start_cash  double precision NOT NULL,
    bench_units double precision NOT NULL,
    cash        double precision NOT NULL,
    crypto      double precision NOT NULL,
    last_candle timestamptz NOT NULL,
    last_buy_at timestamptz,
    last_reason text,
    updated_at  timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS sim_equity (
    simulation text NOT NULL,
    ts         timestamptz NOT NULL,
    price      double precision NOT NULL,
    value      double precision NOT NULL,
    benchmark  double precision NOT NULL,
    action     smallint NOT NULL CHECK (action BETWEEN 0 AND 2),
    q_buy      real,
    q_sell     real,
    q_hold     real,
    PRIMARY KEY (simulation, ts)
);
CREATE TABLE IF NOT EXISTS sim_trades (
    id          bigserial PRIMARY KEY,
    simulation  text NOT NULL,
    ts          timestamptz NOT NULL,
    side        text NOT NULL CHECK (side IN ('buy', 'sell')),
    price       double precision NOT NULL,
    quantity    double precision NOT NULL,
    fee_usdt    double precision NOT NULL,
    value_after double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS sim_trades_simulation_ts ON sim_trades (simulation, ts DESC);
"""

LEGACY_SIMULATION = "dqn-2023"  # the single simulation of the first schema
MIGRATION_V1 = "2026-11-multi-simulations"
LOCK_KEY = 7_381_811  # advisory lock: one process at a time creates or migrates the schema

MIGRATE_V1 = (  # one statement each: parameters cannot go with several statements at once
    """
INSERT INTO sim_portfolio (simulation, config, fee, started_at, start_price, start_cash, bench_units, cash, crypto,
                           last_candle, last_buy_at, last_reason, updated_at)
SELECT %(sim)s,
       jsonb_build_object('strategy', 'dqn', 'params', jsonb_build_object('model', model), 'symbol', symbol,
                          'interval', '1m', 'start_cash', start_cash, 'fee', fee),
       fee, started_at, start_price, start_cash, bench_units, cash, crypto, last_candle,
       (SELECT max(ts) FROM trades WHERE side = 'buy'), NULL, updated_at
FROM portfolio WHERE id = 1""",
    """INSERT INTO sim_equity (simulation, ts, price, value, benchmark, action, q_buy, q_sell, q_hold)
SELECT %(sim)s, ts, price, value, benchmark, action, q_buy, q_sell, q_hold FROM equity""",
    """INSERT INTO sim_trades (simulation, ts, side, price, quantity, fee_usdt, value_after)
SELECT %(sim)s, ts, side, price, quantity, fee_usdt, value_after FROM trades ORDER BY id""",
)

MAX_POINTS = 600  # points of the value series sent to the page
SPARK_POINTS = 48  # points of the small chart on the home page


class ConfigChanged(Exception):
    """The definition in simulations.toml is not the one the stored portfolio started with."""


class StrategyFailed(Exception):
    pass


@dataclass(frozen=True)
class Step:
    ts: datetime
    price: float
    action: int
    value: float
    benchmark: float
    trade: Trade | None
    reason: str


class Store:
    def __init__(self, conninfo: str = "", **params):
        self.conninfo = conninfo
        self.params = {"connect_timeout": 10, "options": "-c statement_timeout=10000 -c TimeZone=UTC", **params}

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.conninfo, row_factory=dict_row, **self.params)

    def init_schema(self) -> bool:
        """Create the tables and run the migration if it has not run yet. True if it migrated now."""
        with self.connect() as conn, conn.transaction():
            conn.execute("SET LOCAL statement_timeout = '300s'")
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
            conn.execute(SCHEMA)
            if conn.execute("SELECT 1 FROM schema_migrations WHERE name = %s", (MIGRATION_V1,)).fetchone():
                return False
            old = conn.execute("SELECT to_regclass('portfolio') IS NOT NULL AND to_regclass('equity') IS NOT NULL"
                               " AND to_regclass('trades') IS NOT NULL AS present").fetchone()["present"]
            if old:
                for statement in MIGRATE_V1:
                    conn.execute(statement, {"sim": LEGACY_SIMULATION})
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (MIGRATION_V1,))
            return bool(old)

    def record(self, sim, ts: datetime, price: float, decide: Callable[[Position], Decision]) -> Step | None:
        """Apply the decision taken on the candle that ended at `ts`. None if that candle was already used.
        decide(position) is called inside the transaction, with the stored balances."""
        with self.connect() as conn, conn.transaction():
            q = "SELECT * FROM sim_portfolio WHERE simulation = %s FOR UPDATE"
            row = conn.execute(q, (sim.id,)).fetchone()
            if row is None:
                bench = buy_and_hold_units(sim.start_cash, price, sim.fee)
                conn.execute(
                    "INSERT INTO sim_portfolio (simulation, config, fee, started_at, start_price, start_cash, bench_units,"
                    " cash, crypto, last_candle, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, %s, now())"
                    " ON CONFLICT (simulation) DO NOTHING",
                    (sim.id, Jsonb(sim.fingerprint), sim.fee, ts, price, sim.start_cash, bench, sim.start_cash,
                     ts - timedelta(milliseconds=sim.interval_ms)))
                row = conn.execute(q, (sim.id,)).fetchone()
            if row["config"] != sim.fingerprint:
                raise ConfigChanged(sim.id)
            if ts <= row["last_candle"]:
                return None
            p = Portfolio(row["cash"], row["crypto"], row["fee"])
            try:
                d = decide(Position(p.cash, p.crypto, row["last_buy_at"]))
            except Exception as e:  # noqa: BLE001 - reported apart from database errors
                raise StrategyFailed(sim.id) from e
            trade = apply(p, d.action, price, d.amount)
            value, benchmark = p.value(price), row["bench_units"] * price
            qv = d.q or (None, None, None)
            conn.execute(
                "INSERT INTO sim_equity (simulation, ts, price, value, benchmark, action, q_buy, q_sell, q_hold)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (sim.id, ts, price, value, benchmark, d.action, *(None if v is None else float(v) for v in qv)))
            if trade is not None:
                conn.execute(
                    "INSERT INTO sim_trades (simulation, ts, side, price, quantity, fee_usdt, value_after)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (sim.id, ts, trade.side, trade.price, trade.quantity, trade.fee_usdt, value))
            last_buy = ts if trade is not None and trade.side == "buy" else row["last_buy_at"]
            conn.execute("UPDATE sim_portfolio SET cash = %s, crypto = %s, last_candle = %s, last_buy_at = %s,"
                         " last_reason = %s, updated_at = now() WHERE simulation = %s",
                         (p.cash, p.crypto, ts, last_buy, d.reason, sim.id))
            return Step(ts, price, d.action, value, benchmark, trade, d.reason)

    @staticmethod
    def _series(conn, sim_id: str, started_at: datetime, since: datetime | None, end: datetime, points: int) -> list:
        start = max(since, started_at) if since else started_at
        step = max(timedelta(minutes=1), (end - start) / points)
        step = timedelta(minutes=-(-step.total_seconds() // 60))  # whole minutes, rounded up
        return conn.execute(
            "SELECT DISTINCT ON (b) date_bin(%s, ts, %s) AS b, ts, price, value, benchmark FROM sim_equity"
            " WHERE simulation = %s AND ts >= %s ORDER BY b, ts DESC", (step, started_at, sim_id, start)).fetchall()

    def snapshot(self, sim_id: str, since: datetime | None, trades_limit: int = 100) -> dict:
        """Everything the page of one simulation shows, in one read. since=None: the whole history."""
        with self.connect() as conn:
            portfolio = conn.execute("SELECT * FROM sim_portfolio WHERE simulation = %s", (sim_id,)).fetchone()
            if portfolio is None:
                return {"portfolio": None, "series": [], "trades": [], "stats": None}
            last = conn.execute("SELECT * FROM sim_equity WHERE simulation = %s ORDER BY ts DESC LIMIT 1",
                                (sim_id,)).fetchone()
            end = last["ts"] if last else portfolio["started_at"]
            series = self._series(conn, sim_id, portfolio["started_at"], since, end, MAX_POINTS)
            trades = conn.execute("SELECT ts, side, price, quantity, fee_usdt, value_after FROM sim_trades"
                                  " WHERE simulation = %s ORDER BY ts DESC, id DESC LIMIT %s",
                                  (sim_id, trades_limit)).fetchall()
            stats = conn.execute(
                "SELECT (SELECT count(*) FROM sim_trades WHERE simulation = %(s)s) AS trades,"
                " (SELECT coalesce(sum(fee_usdt), 0) FROM sim_trades WHERE simulation = %(s)s) AS fees,"
                " (SELECT count(*) FROM sim_equity WHERE simulation = %(s)s) AS decisions,"
                " count(*) FILTER (WHERE action = 0) AS buy_24h, count(*) FILTER (WHERE action = 1) AS sell_24h,"
                " count(*) FILTER (WHERE action = 2) AS hold_24h"
                " FROM sim_equity WHERE simulation = %(s)s AND ts > %(t)s",
                {"s": sim_id, "t": end - timedelta(hours=24)}).fetchone()
            return {"portfolio": portfolio, "last": last, "series": series, "trades": trades, "stats": stats}

    def summary(self, sim_ids: list[str]) -> dict[str, dict]:
        """For the home page: every simulation's portfolio, latest point, trade count and a small series."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT p.*, e.ts AS last_ts, e.price AS last_price,"
                " (SELECT count(*) FROM sim_trades t WHERE t.simulation = p.simulation) AS trades"
                " FROM sim_portfolio p LEFT JOIN LATERAL (SELECT ts, price FROM sim_equity q"
                " WHERE q.simulation = p.simulation ORDER BY ts DESC LIMIT 1) e ON true"
                " WHERE p.simulation = ANY(%s)", (list(sim_ids),)).fetchall()
            out = {}
            for r in rows:
                end = r["last_ts"] or r["started_at"]
                r["spark"] = self._series(conn, r["simulation"], r["started_at"], None, end, SPARK_POINTS)
                out[r["simulation"]] = r
            return out


def utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

