"""Persistence on a real PostgreSQL (see conftest.py)."""

from datetime import datetime, timedelta, timezone

import pytest

from conftest import simulations
from trader.portfolio import BUY, FEE, HOLD, SELL
from trader.store import MAX_POINTS, ConfigChanged, Store, StrategyFailed
from trader.strategies import Decision

T0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
SIMS = {s.id: s for s in simulations("""
[[simulation]]
id = "a"
name = "A"
strategy = "buy-and-hold"
[[simulation]]
id = "b"
name = "B"
strategy = "dca"
interval = "1h"
start_cash = 500
[simulation.params]
amount = 100
every = "2h"
""")}
A, B = SIMS["a"], SIMS["b"]


def rec(store, minute, price, action, sim=A, amount=None, q=None):
    return store.record(sim, T0 + timedelta(minutes=minute), price, lambda pos: Decision(action, f"r{minute}", amount, q))


def test_first_decision_opens_the_portfolio_and_the_benchmark(store):
    step = rec(store, 0, 50_000.0, HOLD)
    assert step.trade is None and step.reason == "r0"
    assert step.value == pytest.approx(1000.0)
    assert step.benchmark == pytest.approx(1000.0 * (1 - FEE))
    snap = store.snapshot("a", None)
    p = snap["portfolio"]
    assert (p["cash"], p["crypto"], p["start_price"], p["started_at"]) == (1000.0, 0.0, 50_000.0, T0)
    assert p["config"] == A.fingerprint and p["last_reason"] == "r0"
    assert snap["last"]["q_buy"] is None


def test_trades_and_fees_are_recorded(store):
    rec(store, 0, 50_000.0, BUY, q=(0.3, 0.2, 0.1))
    rec(store, 1, 51_000.0, HOLD)
    step = rec(store, 2, 52_000.0, SELL)
    assert step.trade.side == "sell"
    expected_cash = 1000 * (1 - FEE) / 50_000 * 52_000 * (1 - FEE)
    assert step.value == pytest.approx(expected_cash)
    snap = store.snapshot("a", None)
    assert [t["side"] for t in snap["trades"]] == ["sell", "buy"]  # latest first
    assert snap["stats"]["trades"] == 2
    assert snap["stats"]["fees"] == pytest.approx(1.0 + 52_000 * (1000 * (1 - FEE) / 50_000) * FEE)
    assert snap["stats"]["decisions"] == 3
    assert (snap["stats"]["buy_24h"], snap["stats"]["sell_24h"], snap["stats"]["hold_24h"]) == (1, 1, 1)
    assert snap["portfolio"]["cash"] == pytest.approx(expected_cash) and snap["portfolio"]["crypto"] == 0.0
    assert snap["portfolio"]["last_buy_at"] == T0


def test_simulations_are_independent(store):
    rec(store, 0, 50_000.0, BUY)
    rec(store, 60, 40_000.0, BUY, sim=B, amount=100)
    a, b = store.snapshot("a", None), store.snapshot("b", None)
    assert a["portfolio"]["cash"] == 0.0 and b["portfolio"]["cash"] == 400.0
    assert b["portfolio"]["start_cash"] == 500.0 and b["portfolio"]["started_at"] == T0 + timedelta(hours=1)
    assert b["portfolio"]["bench_units"] == pytest.approx(500 / 40_000 * (1 - FEE))
    assert a["stats"]["trades"] == b["stats"]["trades"] == 1
    summary = store.summary(["a", "b", "missing"])
    assert set(summary) == {"a", "b"}
    assert summary["b"]["trades"] == 1 and summary["b"]["last_price"] == 40_000.0 and len(summary["b"]["spark"]) == 1


def test_the_strategy_sees_the_stored_position(store):
    seen = []

    def decide(pos):
        seen.append(pos)
        return Decision(BUY, "dca", 100)

    store.record(B, T0, 40_000.0, decide)
    store.record(B, T0 + timedelta(hours=1), 41_000.0, decide)
    assert seen[0].cash == 500.0 and seen[0].last_buy_at is None
    assert seen[1].cash == 400.0 and seen[1].crypto > 0 and seen[1].last_buy_at == T0


def test_a_candle_is_used_once(store):
    assert rec(store, 0, 50_000.0, BUY) is not None
    assert rec(store, 0, 50_000.0, SELL) is None  # same candle again (restart, retry)
    assert rec(store, -5, 49_000.0, SELL) is None  # older candle
    snap = store.snapshot("a", None)
    assert snap["stats"]["decisions"] == 1 and snap["stats"]["trades"] == 1


def test_a_failing_strategy_writes_nothing(store):
    rec(store, 0, 50_000.0, HOLD)

    def broken(pos):
        raise ZeroDivisionError

    with pytest.raises(StrategyFailed):
        store.record(A, T0 + timedelta(minutes=1), 50_000.0, broken)
    snap = store.snapshot("a", None)
    assert snap["stats"]["decisions"] == 1 and snap["portfolio"]["last_candle"] == T0


def test_a_changed_definition_is_refused(store):
    rec(store, 0, 50_000.0, HOLD)
    (changed,) = simulations('[[simulation]]\nid = "a"\nname = "A, renamed"\nstrategy = "buy-and-hold"\nfee = 0.002\n')
    with pytest.raises(ConfigChanged):
        store.record(changed, T0 + timedelta(minutes=1), 50_000.0, lambda pos: Decision(HOLD, ""))
    (renamed,) = simulations('[[simulation]]\nid = "a"\nname = "A, renamed"\ndescription = "new"\nstrategy = "buy-and-hold"\n')
    assert store.record(renamed, T0 + timedelta(minutes=1), 50_000.0, lambda pos: Decision(HOLD, "")) is not None


def test_state_survives_a_restart(store, pg_url):
    rec(store, 0, 50_000.0, BUY)
    again = Store(pg_url, connect_timeout=30)  # a new process, same database
    assert again.init_schema() is False  # idempotent
    step = again.record(A, T0 + timedelta(minutes=1), 55_000.0, lambda pos: Decision(HOLD, ""))
    assert step.value == pytest.approx(1000 * (1 - FEE) / 50_000 * 55_000)
    assert again.snapshot("a", None)["portfolio"]["started_at"] == T0


def test_series_is_downsampled_and_keeps_the_latest_point(store):
    rec(store, 0, 50_000.0, HOLD)
    with store.connect() as conn, conn.transaction():
        conn.execute(
            "INSERT INTO sim_equity (simulation, ts, price, value, benchmark, action)"
            " SELECT 'a', %s + make_interval(mins => g), 50000 + g, 1000, 999 + g, 2"
            " FROM generate_series(1, 3000) g", (T0,))
        conn.execute("INSERT INTO sim_equity (simulation, ts, price, value, benchmark, action)"
                     " VALUES ('b', %s, 1, 1, 1, 2)", (T0 + timedelta(days=9),))  # another simulation: not mixed in
    series = store.snapshot("a", None)["series"]
    assert len(series) <= MAX_POINTS + 1
    assert series[-1]["ts"] == T0 + timedelta(minutes=3000)
    assert all(x["ts"] < y["ts"] for x, y in zip(series, series[1:]))
    day = store.snapshot("a", T0 + timedelta(minutes=3000) - timedelta(days=1))["series"]
    assert day[0]["ts"] >= T0 + timedelta(minutes=3000) - timedelta(days=1)
    assert len(store.summary(["a"])["a"]["spark"]) <= 49


def test_empty_database(store):
    assert store.snapshot("a", None)["portfolio"] is None
    assert store.summary(["a"]) == {}
