"""Every strategy on synthetic candles: the signal it gives, and the same accounting for all of them."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from conftest import MODEL
from trader import features
from trader.market import Candles, INTERVALS
from trader.model import QNetwork
from trader.portfolio import BUY, FEE, HOLD, SELL, Portfolio, apply, buy_and_hold_units
from trader.strategies import ConfigError, Position, build

T0 = 1_795_000_000_000 // 3_600_000 * 3_600_000  # a whole hour, in ms
CASH = Position(1000.0, 0.0)
COIN = Position(0.0, 0.02)


def candles(close, interval="1h", volume=None):
    ms = INTERVALS[interval]
    close = [float(c) for c in close]
    return Candles([T0 + i * ms for i in range(len(close))], close, volume or [1.0] * len(close), ms)


def ramp(n, start=60_000.0, step=10.0):
    return [start + i * step for i in range(n)]


def test_sma_cross_follows_the_trend():
    s = build("sma-cross", {"fast": 5, "slow": 20})
    assert s.lookback == 20
    assert s.decide(candles(ramp(20)), CASH).action == BUY  # rising: fast average above the slow one
    d = s.decide(candles(ramp(20, step=-10)), COIN)
    assert d.action == SELL and "below SMA 20" in d.reason
    assert s.decide(candles([100] * 20), CASH).action == HOLD


def test_sma_cross_uses_the_latest_candles_only():
    s = build("sma-cross", {"fast": 2, "slow": 3})
    # old history falling, the last three rising: only the last three count
    assert s.decide(candles([500, 400, 300, 1, 2, 3]), CASH).action == BUY


def test_rsi_buys_oversold_and_sells_overbought():
    s = build("rsi", {"period": 14, "oversold": 30, "overbought": 70})
    assert s.lookback == 280
    rng = np.random.default_rng(5)
    falling = 60_000 * np.exp(np.cumsum(rng.normal(-0.003, 0.001, 300)))
    rising = 60_000 * np.exp(np.cumsum(rng.normal(0.003, 0.001, 300)))
    flat = 60_000 * np.exp(np.cumsum(np.tile([0.001, -0.001], 150)))
    d = s.decide(candles(falling), CASH)
    assert d.action == BUY and "RSI 14 at" in d.reason
    assert s.decide(candles(rising), COIN).action == SELL
    assert s.decide(candles(flat), CASH).action == HOLD


def test_rsi_matches_ta():
    ta = pytest.importorskip("ta")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(9)
    close = 60_000 * np.exp(np.cumsum(rng.normal(0, 0.002, 400)))
    for period in (7, 14, 21):
        ours = features.rsi(close, period)
        ref = ta.momentum.RSIIndicator(pd.Series(close), window=period).rsi().to_numpy()
        np.testing.assert_allclose(ours[period:], ref[period:], rtol=1e-9)


def test_bollinger_bands():
    s = build("bollinger", {"period": 20, "width": 2.0})
    base = [100.0, 101.0] * 10
    assert s.decide(candles(base[:-1] + [90.0]), CASH).action == BUY  # far below the lower band
    assert s.decide(candles(base[:-1] + [112.0]), COIN).action == SELL
    d = s.decide(candles(base), CASH)
    assert d.action == HOLD and "inside the bands" in d.reason
    close = np.array(base[:-1] + [90.0])
    mid, sd = close.mean(), close.std()
    assert f"{mid - 2 * sd:,.2f}" in s.decide(candles(close), CASH).reason


def test_dca_buys_a_fixed_amount_on_schedule():
    s = build("dca", {"amount": 25, "every": "1d"})
    c = candles([100.0] * 3)  # the latest candle ends at T0 + 3h
    end = datetime.fromtimestamp(c.last_close_time_ms / 1000, tz=timezone.utc)
    first = s.decide(c, CASH)
    assert (first.action, first.amount) == (BUY, 25.0)
    assert s.decide(c, Position(975.0, 0.25, end - timedelta(hours=23))).action == HOLD
    assert s.decide(c, Position(975.0, 0.25, end - timedelta(days=1))).action == BUY
    assert s.decide(c, Position(0.0, 10.0, end - timedelta(days=5))).action == HOLD  # nothing left to spend


def test_buy_and_hold():
    s = build("buy-and-hold", {})
    assert s.decide(candles([100.0]), CASH).action == BUY
    assert s.decide(candles([100.0]), COIN).action == HOLD


def test_dqn_is_the_2023_pipeline():
    """Same input, same model, same decision as the single-simulation service: the 499 closed candles of a
    500-candle request, the observation of features.py and the greedy action of the network."""
    s = build("dqn", {}, model_dir=MODEL.parent)
    assert s.lookback == 499
    rng = np.random.default_rng(3)
    close = list(60_000 * np.exp(np.cumsum(rng.normal(0, 0.001, 600))))
    volume = list(rng.gamma(2.0, 4.0, 600))
    c = candles(close, "1m", volume).tail(s.lookback)
    d = s.decide(c, CASH)
    action, q = QNetwork(str(MODEL)).decide(features.observation(np.array(close[-499:]), np.array(volume[-499:])))
    assert d.action == action and d.q == tuple(float(v) for v in q)


def test_dqn_needs_its_model_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        build("dqn", {"model": "missing.h5"}, model_dir=tmp_path)
    with pytest.raises(ConfigError, match="not found"):
        build("dqn", {"model": "../completed/trading_model.h5"}, model_dir=MODEL.parent)


@pytest.mark.parametrize("kind,params,message", [
    ("sma-cross", {"fast": 50, "slow": 20}, "fast must be smaller"),
    ("sma-cross", {"fast": 1}, "between"),
    ("sma-cross", {"fast": 2.5}, "must be an integer"),
    ("sma-cross", {"fats": 5}, "unknown parameter fats"),
    ("rsi", {"oversold": 80, "overbought": 70}, "oversold must be smaller"),
    ("rsi", {"period": 60}, "between"),
    ("rsi", {"oversold": "low"}, "must be a number"),
    ("bollinger", {"width": 0}, "between"),
    ("dca", {"every": "daily"}, "duration"),
    ("dca", {"amount": -5}, "between"),
    ("buy-and-hold", {"x": 1}, "unknown parameter"),
    ("martingale", {}, "unknown strategy"),
])
def test_bad_parameters_are_refused(kind, params, message):
    with pytest.raises(ConfigError, match=message):
        build(kind, params)


def replay(strategy, close, fee=FEE, interval="1h"):
    """Run a strategy candle by candle over a price series with the service's accounting."""
    p, last_buy, trades = Portfolio(1000.0, 0.0, fee), None, []
    ms = INTERVALS[interval]
    for i in range(strategy.lookback, len(close) + 1):
        c = candles(close[:i], interval).tail(strategy.lookback)
        d = strategy.decide(c, Position(p.cash, p.crypto, last_buy))
        t = apply(p, d.action, close[i - 1], d.amount)
        if t:
            trades.append(t)
            if t.side == "buy":
                last_buy = datetime.fromtimestamp((T0 + i * ms) / 1000, tz=timezone.utc)
    return p, trades


@pytest.mark.parametrize("kind,params", [("sma-cross", {"fast": 5, "slow": 20}), ("rsi", {"period": 7}),
                                         ("bollinger", {"period": 10, "width": 1.5}),
                                         ("dca", {"amount": 30, "every": "6h"}), ("buy-and-hold", {})])
def test_same_accounting_for_every_strategy(kind, params):
    rng = np.random.default_rng(11)
    close = list(60_000 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    p, trades = replay(build(kind, params), close)
    assert trades, "the synthetic market should make every strategy trade"
    bought = sum(t.quantity for t in trades if t.side == "buy")
    sold = sum(t.quantity for t in trades if t.side == "sell")
    assert p.crypto == pytest.approx(bought - sold, abs=1e-12)
    spent = sum(t.quantity * t.price / (1 - FEE) for t in trades if t.side == "buy")  # gross USDT of the buys
    received = sum(t.quantity * t.price - t.fee_usdt for t in trades if t.side == "sell")
    assert p.cash == pytest.approx(1000.0 - spent + received, rel=1e-9)
    assert all(t.fee_usdt == pytest.approx(t.quantity * t.price * FEE / (1 - FEE) if t.side == "buy"
                                           else t.quantity * t.price * FEE) for t in trades)
    assert p.cash >= 0 and p.crypto >= 0


def test_buy_and_hold_simulation_is_its_own_benchmark():
    rng = np.random.default_rng(2)
    close = list(60_000 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))))
    p, trades = replay(build("buy-and-hold", {}), close)
    assert len(trades) == 1
    assert p.value(close[-1]) == pytest.approx(buy_and_hold_units(1000.0, close[0]) * close[-1])


def test_dca_spends_the_cash_in_equal_parts():
    close = [60_000.0] * 200
    p, trades = replay(build("dca", {"amount": 100, "every": "12h"}), close)
    assert len(trades) == 10 and p.cash == 0.0  # 1000 USDT in ten buys of 100, then nothing left
    assert p.crypto == pytest.approx(1000 / 60_000 * (1 - FEE))
