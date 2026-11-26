"""The risk statistics on synthetic series whose answers are known in advance."""

import math

import numpy as np
import pytest

from trader.metrics import DAY_MS, MIN_RETURNS, of_series, periods_per_year

HOUR = 3_600_000
T0 = 1_767_225_600_000  # 2026-01-01 00:00 UTC


def hours(n):
    return [T0 + i * HOUR for i in range(n)]


def from_returns(returns, start=1000.0):
    return list(start * np.cumprod(1 + np.asarray(returns)))


def test_candles_in_a_year():
    assert periods_per_year(60_000) == 525_600
    assert periods_per_year(HOUR) == 8_760
    assert periods_per_year(DAY_MS) == 365


def test_steady_growth_has_no_volatility_and_no_drawdown():
    m = of_series(hours(30 * 24), from_returns([0.001] * (30 * 24)), HOUR, start_value=1000.0)
    assert m["enough"] is True
    assert m["volatility"] == pytest.approx(0, abs=1e-12)
    assert m["annual_return"] == pytest.approx(0.001 * 8760)
    assert m["sharpe"] is None  # a return without risk: the ratio has no meaning
    assert m["max_drawdown"] == 0 and m["max_drawdown_at"] is None


def test_volatility_of_alternating_returns():
    n = 10 * 24  # even: the mean is exactly zero
    r = [0.01, -0.01] * (n // 2)
    m = of_series(hours(n), from_returns(r), HOUR, start_value=1000.0)
    std = 0.01 * math.sqrt(n / (n - 1))  # sample standard deviation of +-1%
    assert m["volatility"] == pytest.approx(std * math.sqrt(8760))
    assert m["annual_return"] == pytest.approx(0, abs=1e-12)
    assert m["sharpe"] == pytest.approx(0, abs=1e-9)


def test_sharpe_is_the_annualized_mean_over_the_volatility():
    n = 14
    r = [0.02, 0.0] * (n // 2)  # mean 1%, sample standard deviation 1% x sqrt(n / (n - 1)), one candle a day
    days = [T0 + i * DAY_MS for i in range(n)]
    assert of_series(days, from_returns(r), DAY_MS, start_value=1000.0)["enough"] is False  # only 14 candles
    n = 60
    r = [0.02, 0.0] * (n // 2)
    days = [T0 + i * DAY_MS for i in range(n)]
    m = of_series(days, from_returns(r), DAY_MS, start_value=1000.0)
    std = 0.01 * math.sqrt(n / (n - 1))
    assert m["volatility"] == pytest.approx(std * math.sqrt(365))
    assert m["annual_return"] == pytest.approx(0.01 * 365)
    assert m["sharpe"] == pytest.approx(0.01 / std * math.sqrt(365))


def test_maximum_drawdown_and_its_date():
    # up to 1200, down to 900 (-25% from the peak), up to 1300, down to 1100 (only -15%): the deepest is -25%
    v = [1000 + 10 * i for i in range(21)]           # 1000 .. 1200, hours 0..20
    v += [1200 - 15 * i for i in range(1, 21)]       # .. 900 at hour 40
    v += [900 + 20 * i for i in range(1, 21)]        # .. 1300 at hour 60
    v += [1300 - 10 * i for i in range(1, 21)]       # .. 1100 at hour 80
    v += [1100] * (8 * 24 - len(v))
    t = hours(len(v))
    m = of_series(t, v, HOUR, start_value=1000.0)
    assert m["max_drawdown"] == pytest.approx(900 / 1200 - 1)
    assert m["max_drawdown_at"] == t[40]


def test_the_starting_cash_is_the_first_peak():
    v = [999.0 - i for i in range(8 * 24)]  # never above the start: the fall is measured from 1000
    t = hours(len(v))
    m = of_series(t, v, HOUR, start_value=1000.0)
    assert m["max_drawdown"] == pytest.approx(v[-1] / 1000 - 1)
    assert m["max_drawdown_at"] == t[-1]


def test_not_enough_data_yet():
    six_days = of_series(hours(6 * 24), from_returns([0.001, -0.002] * (3 * 24)), HOUR, start_value=1000.0)
    assert six_days["enough"] is False
    assert six_days["volatility"] is None and six_days["sharpe"] is None and six_days["max_drawdown"] is None
    week = of_series(hours(7 * 24), from_returns([0.001, -0.002] * (7 * 12)), HOUR, start_value=1000.0)
    assert week["enough"] is True  # 7 days with the starting point
    few = of_series([T0 + i * DAY_MS for i in range(10)], [1000.0 + i for i in range(10)], DAY_MS, start_value=1000.0)
    assert few["days"] >= 7 and few["returns"] < MIN_RETURNS and few["enough"] is False
    assert of_series([T0], [1000.0], HOUR)["enough"] is False
