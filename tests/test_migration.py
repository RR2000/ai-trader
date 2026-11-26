"""The single-simulation database of the first release becomes simulation dqn-2023 without losing anything.

The old database is written by the store of that release (tests/store_v1.py, frozen), as production did."""

from datetime import datetime, timedelta, timezone

import pytest

import store_v1
from conftest import MODEL
from trader.config import load
from trader.service import ROOT
from trader.store import LEGACY_SIMULATION, Store
from trader.strategies import Decision

T0 = datetime(2026, 11, 25, 22, 35, tzinfo=timezone.utc)
DQN = next(s for s in load(ROOT / "simulations.toml", model_dir=MODEL.parent, networks={}) if s.id == LEGACY_SIMULATION)


@pytest.fixture
def old(pg_url, store):
    """A database as production had it: the three v1 tables only, ten hours of decisions and some trades."""
    with store.connect() as conn:
        conn.execute("DROP TABLE IF EXISTS schema_migrations, sim_portfolio, sim_equity, sim_trades")
    v1 = store_v1.Store(pg_url, connect_timeout=30)
    v1.init_schema()
    actions = [0, 2, 2, 1, 2, 0, 0, 1, 1, 2] * 4  # one connection per decision: slow on Windows, so 40 of them
    for i, a in enumerate(actions):
        v1.record(T0 + timedelta(minutes=i), 60_000.0 + 37 * ((i * 7) % 13) - 5 * i % 11, a, (0.1 * a, 0.2, 0.3 - i / 1e4),
                  symbol="BTCUSDT", model="trading_model_episode_30.h5")
    with v1.connect() as conn, conn.transaction():  # the remaining minutes in bulk, without trades
        conn.execute("INSERT INTO equity SELECT ts + interval '1 minute' * g, price, value, benchmark, 2, q_buy, q_sell,"
                     " q_hold FROM equity, generate_series(1, 566) g WHERE ts = (SELECT max(ts) FROM equity)"
                     " ON CONFLICT DO NOTHING")
        conn.execute("UPDATE portfolio SET last_candle = (SELECT max(ts) FROM equity)")
    return v1


def dump(conn, sql):
    return [tuple(r.values()) for r in conn.execute(sql).fetchall()]


def test_migration_copies_everything(old, pg_url):
    with old.connect() as conn:
        before = {
            "p": dump(conn, "SELECT fee, started_at, start_price, start_cash, bench_units, cash, crypto, last_candle,"
                            " updated_at FROM portfolio"),
            "e": dump(conn, "SELECT ts, price, value, benchmark, action, q_buy, q_sell, q_hold FROM equity ORDER BY ts"),
            "t": dump(conn, "SELECT ts, side, price, quantity, fee_usdt, value_after FROM trades ORDER BY id"),
        }
    assert len(before["e"]) == 606 and len(before["t"]) >= 10
    store = Store(pg_url, connect_timeout=30)
    assert store.init_schema() is True
    with store.connect() as conn:
        after = {
            "p": dump(conn, "SELECT fee, started_at, start_price, start_cash, bench_units, cash, crypto, last_candle,"
                            " updated_at FROM sim_portfolio WHERE simulation = 'dqn-2023'"),
            "e": dump(conn, "SELECT ts, price, value, benchmark, action, q_buy, q_sell, q_hold FROM sim_equity"
                            " WHERE simulation = 'dqn-2023' ORDER BY ts"),
            "t": dump(conn, "SELECT ts, side, price, quantity, fee_usdt, value_after FROM sim_trades"
                            " WHERE simulation = 'dqn-2023' ORDER BY id"),
        }
        config = conn.execute("SELECT config, last_buy_at FROM sim_portfolio").fetchone()
        old_tables = dump(conn, "SELECT (SELECT count(*) FROM portfolio) AS p, (SELECT count(*) FROM equity) AS e,"
                                " (SELECT count(*) FROM trades) AS t")
    assert after == before  # every row, every value, bit for bit
    assert old_tables == [(1, 606, len(before["t"]))]  # the old tables are left as they were
    assert config["config"] == DQN.fingerprint  # so simulations.toml's dqn-2023 continues it
    assert config["last_buy_at"] == max(t[0] for t in before["t"] if t[1] == "buy")

    # idempotent: a second start (or a second replica) migrates nothing again
    assert store.init_schema() is False
    with store.connect() as conn:
        assert dump(conn, "SELECT count(*) FROM sim_equity") == [(606,)]
        assert dump(conn, "SELECT count(*) FROM sim_trades") == [(len(before["t"]),)]


def test_dqn_2023_continues_where_it_was(old, pg_url):
    with old.connect() as conn:
        p = conn.execute("SELECT * FROM portfolio").fetchone()
        trades = conn.execute("SELECT count(*) AS n FROM trades").fetchone()["n"]
    store = Store(pg_url, connect_timeout=30)
    store.init_schema()
    assert store.record(DQN, p["last_candle"], 1.0, lambda pos: Decision(0, "")) is None  # already processed
    seen = []
    step = store.record(DQN, p["last_candle"] + timedelta(minutes=1), 61_000.0,
                        lambda pos: seen.append(pos) or Decision(1 if pos.crypto else 0, "x", q=(0.1, 0.2, 0.3)))
    assert (seen[0].cash, seen[0].crypto) == (p["cash"], p["crypto"])
    assert step.benchmark == pytest.approx(p["bench_units"] * 61_000.0)
    snap = store.snapshot(LEGACY_SIMULATION, None)
    assert snap["portfolio"]["started_at"] == T0 == p["started_at"]
    assert snap["stats"]["decisions"] == 607 and snap["stats"]["trades"] == trades + 1


def test_a_new_database_has_nothing_to_migrate(store):
    with store.connect() as conn:
        assert dump(conn, "SELECT count(*) FROM sim_portfolio") == [(0,)]
        assert dump(conn, "SELECT name FROM schema_migrations") == [("2026-11-multi-simulations",)]
        assert conn.execute("SELECT to_regclass('portfolio') AS t").fetchone()["t"] is None
    assert store.init_schema() is False


def test_a_failed_migration_changes_nothing(old, pg_url):
    store = Store(pg_url, connect_timeout=30)
    with store.connect() as conn:  # a row in the way: the copy fails half-way
        conn.execute("CREATE TABLE sim_trades (id bigserial PRIMARY KEY, simulation text NOT NULL, ts timestamptz NOT NULL,"
                     " side text NOT NULL CHECK (side = 'sell'), price double precision NOT NULL, quantity double"
                     " precision NOT NULL, fee_usdt double precision NOT NULL, value_after double precision NOT NULL)")
    with pytest.raises(Exception):
        store.init_schema()
    with store.connect() as conn:
        assert conn.execute("SELECT to_regclass('sim_portfolio') AS t").fetchone()["t"] is None  # rolled back
        assert conn.execute("SELECT count(*) AS n FROM equity").fetchone()["n"] == 606
        conn.execute("DROP TABLE sim_trades")
    assert store.init_schema() is True  # and it runs once the obstacle is gone
