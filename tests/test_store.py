"""Persistence on a real PostgreSQL (see conftest.py)."""

from datetime import datetime, timedelta, timezone

import pytest

from trader.portfolio import FEE
from trader.store import MAX_POINTS, Store

T0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
Q = (0.1, 0.2, 0.3)


def rec(store, minute, price, action):
    return store.record(T0 + timedelta(minutes=minute), price, action, Q, symbol="BTCUSDT", model="m.h5")


def test_first_decision_opens_the_portfolio_and_the_benchmark(store):
    step = rec(store, 0, 50_000.0, 2)
    assert step.trade is None
    assert step.value == pytest.approx(1000.0)
    assert step.benchmark == pytest.approx(1000.0 * (1 - FEE))
    snap = store.snapshot(None)
    p = snap["portfolio"]
    assert (p["cash"], p["crypto"], p["start_price"], p["started_at"]) == (1000.0, 0.0, 50_000.0, T0)


def test_trades_and_fees_are_recorded(store):
    rec(store, 0, 50_000.0, 0)  # buy
    rec(store, 1, 51_000.0, 2)  # hold
    step = rec(store, 2, 52_000.0, 1)  # sell
    assert step.trade.side == "sell"
    expected_cash = 1000 * (1 - FEE) / 50_000 * 52_000 * (1 - FEE)
    assert step.value == pytest.approx(expected_cash)
    snap = store.snapshot(None)
    assert [t["side"] for t in snap["trades"]] == ["sell", "buy"]  # latest first
    assert snap["stats"]["trades"] == 2
    assert snap["stats"]["fees"] == pytest.approx(1.0 + 52_000 * (1000 * (1 - FEE) / 50_000) * FEE)
    assert snap["stats"]["decisions"] == 3
    assert (snap["stats"]["buy_24h"], snap["stats"]["sell_24h"], snap["stats"]["hold_24h"]) == (1, 1, 1)
    assert snap["portfolio"]["cash"] == pytest.approx(expected_cash) and snap["portfolio"]["crypto"] == 0.0


def test_a_candle_is_used_once(store):
    assert rec(store, 0, 50_000.0, 0) is not None
    assert rec(store, 0, 50_000.0, 1) is None  # same candle again (restart, retry)
    assert rec(store, -5, 49_000.0, 1) is None  # older candle
    snap = store.snapshot(None)
    assert snap["stats"]["decisions"] == 1 and snap["stats"]["trades"] == 1


def test_state_survives_a_restart(store, pg_url):
    rec(store, 0, 50_000.0, 0)
    again = Store(pg_url, connect_timeout=30)  # a new process, same database
    again.init_schema()  # idempotent
    step = again.record(T0 + timedelta(minutes=1), 55_000.0, 2, Q, symbol="BTCUSDT", model="m.h5")
    assert step.value == pytest.approx(1000 * (1 - FEE) / 50_000 * 55_000)
    assert again.snapshot(None)["portfolio"]["started_at"] == T0


def test_series_is_downsampled_and_keeps_the_latest_point(store):
    with store.connect() as conn, conn.transaction():
        rec(store, 0, 50_000.0, 2)
        conn.execute(
            "INSERT INTO equity (ts, price, value, benchmark, action, q_buy, q_sell, q_hold)"
            " SELECT %s + make_interval(mins => g), 50000 + g, 1000, 999 + g, 2, 0, 0, 0"
            " FROM generate_series(1, 3000) g", (T0,))
    series = store.snapshot(None)["series"]
    assert len(series) <= MAX_POINTS + 1
    assert series[-1]["ts"] == T0 + timedelta(minutes=3000)
    assert all(a["ts"] < b["ts"] for a, b in zip(series, series[1:]))
    day = store.snapshot(T0 + timedelta(minutes=3000) - timedelta(days=1))["series"]
    assert day[0]["ts"] >= T0 + timedelta(minutes=3000) - timedelta(days=1)


def test_empty_database(store):
    assert store.snapshot(None)["portfolio"] is None
