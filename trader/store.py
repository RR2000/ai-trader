"""Persistence on PostgreSQL: the portfolio, one row per decision (the value series) and the trades.

The state lives in the database, so a restart or a new deploy continues the same portfolio.
Every decision is one transaction: the equity row, the trade (if any) and the new balances
are written together or not at all, and a candle already processed is never applied twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg.rows import dict_row

from .portfolio import FEE, START_CASH, Portfolio, Trade, apply, buy_and_hold_units

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio (
    id          smallint PRIMARY KEY CHECK (id = 1),
    symbol      text NOT NULL,
    model       text NOT NULL,
    fee         double precision NOT NULL,
    started_at  timestamptz NOT NULL,
    start_price double precision NOT NULL,
    start_cash  double precision NOT NULL,
    bench_units double precision NOT NULL,
    cash        double precision NOT NULL,
    crypto      double precision NOT NULL,
    last_candle timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS equity (
    ts        timestamptz PRIMARY KEY,
    price     double precision NOT NULL,
    value     double precision NOT NULL,
    benchmark double precision NOT NULL,
    action    smallint NOT NULL CHECK (action BETWEEN 0 AND 2),
    q_buy     real NOT NULL,
    q_sell    real NOT NULL,
    q_hold    real NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id          bigserial PRIMARY KEY,
    ts          timestamptz NOT NULL,
    side        text NOT NULL CHECK (side IN ('buy', 'sell')),
    price       double precision NOT NULL,
    quantity    double precision NOT NULL,
    fee_usdt    double precision NOT NULL,
    value_after double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS trades_ts ON trades (ts DESC);
"""

MAX_POINTS = 600  # points of the value series sent to the page


@dataclass(frozen=True)
class Step:
    ts: datetime
    price: float
    action: int
    value: float
    benchmark: float
    trade: Trade | None


class Store:
    def __init__(self, conninfo: str = "", **params):
        self.conninfo = conninfo
        self.params = {"connect_timeout": 10, "options": "-c statement_timeout=10000 -c TimeZone=UTC", **params}

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.conninfo, row_factory=dict_row, **self.params)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(SCHEMA)

    def record(self, ts: datetime, price: float, action: int, q: tuple[float, float, float], *,
               symbol: str, model: str, start_cash: float = START_CASH, fee: float = FEE) -> Step | None:
        """Apply one decision taken on the candle that ended at `ts`. None if that candle was already used."""
        with self.connect() as conn, conn.transaction():
            row = conn.execute("SELECT * FROM portfolio WHERE id = 1 FOR UPDATE").fetchone()
            if row is None:
                bench = buy_and_hold_units(start_cash, price, fee)
                conn.execute(
                    "INSERT INTO portfolio (id, symbol, model, fee, started_at, start_price, start_cash, bench_units,"
                    " cash, crypto, last_candle, updated_at) VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, now())"
                    " ON CONFLICT (id) DO NOTHING",
                    (symbol, model, fee, ts, price, start_cash, bench, start_cash, ts - timedelta(minutes=1)))
                row = conn.execute("SELECT * FROM portfolio WHERE id = 1 FOR UPDATE").fetchone()
            if ts <= row["last_candle"]:
                return None
            p = Portfolio(row["cash"], row["crypto"], row["fee"])
            trade = apply(p, action, price)
            value, benchmark = p.value(price), row["bench_units"] * price
            conn.execute(
                "INSERT INTO equity (ts, price, value, benchmark, action, q_buy, q_sell, q_hold)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (ts, price, value, benchmark, action, float(q[0]), float(q[1]), float(q[2])))
            if trade is not None:
                conn.execute(
                    "INSERT INTO trades (ts, side, price, quantity, fee_usdt, value_after) VALUES (%s, %s, %s, %s, %s, %s)",
                    (ts, trade.side, trade.price, trade.quantity, trade.fee_usdt, value))
            conn.execute("UPDATE portfolio SET cash = %s, crypto = %s, last_candle = %s, updated_at = now() WHERE id = 1",
                         (p.cash, p.crypto, ts))
            return Step(ts, price, action, value, benchmark, trade)

    def snapshot(self, since: datetime | None, trades_limit: int = 100) -> dict:
        """Everything the page shows, in one read. since=None: the whole history."""
        with self.connect() as conn:
            portfolio = conn.execute("SELECT * FROM portfolio WHERE id = 1").fetchone()
            if portfolio is None:
                return {"portfolio": None, "series": [], "trades": [], "stats": None}
            last = conn.execute("SELECT * FROM equity ORDER BY ts DESC LIMIT 1").fetchone()
            start = max(since, portfolio["started_at"]) if since else portfolio["started_at"]
            end = last["ts"] if last else start
            step = max(timedelta(minutes=1), (end - start) / MAX_POINTS)
            step = timedelta(minutes=-(-step.total_seconds() // 60))  # whole minutes, rounded up
            series = conn.execute(
                "SELECT DISTINCT ON (b) date_bin(%s, ts, %s) AS b, ts, price, value, benchmark FROM equity"
                " WHERE ts >= %s ORDER BY b, ts DESC", (step, portfolio["started_at"], start)).fetchall()
            trades = conn.execute("SELECT ts, side, price, quantity, fee_usdt, value_after FROM trades"
                                  " ORDER BY ts DESC LIMIT %s", (trades_limit,)).fetchall()
            stats = conn.execute(
                "SELECT (SELECT count(*) FROM trades) AS trades, (SELECT coalesce(sum(fee_usdt), 0) FROM trades) AS fees,"
                " (SELECT count(*) FROM equity) AS decisions,"
                " count(*) FILTER (WHERE action = 0) AS buy_24h, count(*) FILTER (WHERE action = 1) AS sell_24h,"
                " count(*) FILTER (WHERE action = 2) AS hold_24h"
                " FROM equity WHERE ts > %s", (end - timedelta(hours=24),)).fetchone()
            return {"portfolio": portfolio, "last": last, "series": series, "trades": trades, "stats": stats}


def utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
