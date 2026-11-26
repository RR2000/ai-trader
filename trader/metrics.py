"""Risk statistics of a value series: annualized volatility, maximum drawdown and Sharpe ratio.

The same formulas for the live simulations (the database computes the sums, see Store.risk) and for the
walk-forward test of training/train_mlp.py (of_series, on the whole series in memory):

- returns per candle: r = v[t] / v[t-1] - 1, the first one from the starting cash;
- annualized volatility: sample standard deviation of r x sqrt(candles in a year), a year being 365 days
  (crypto trades every day);
- annualized return: mean of r x candles in a year;
- Sharpe ratio: annualized return / annualized volatility, risk-free rate 0;
- maximum drawdown: the largest fall from a previous peak (the starting cash counts as one), with its low.

With less than MIN_DAYS of history or fewer than MIN_RETURNS candles the numbers mean nothing yet:
finish() then returns enough=False and no numbers.
"""

from __future__ import annotations

import math

import numpy as np

DAY_MS = 86_400_000
YEAR_MS = 365 * DAY_MS
MIN_DAYS = 7
MIN_RETURNS = 30


def periods_per_year(interval_ms: int) -> float:
    return YEAR_MS / interval_ms


def finish(n: int, mean: float | None, std: float | None, drawdown: float | None, drawdown_at, span_ms: float,
           interval_ms: int) -> dict:
    """The statistics from n returns per candle (their mean and sample standard deviation), the deepest fall
    (a fraction, 0 or negative) and when it hit bottom, over span_ms of history."""
    out = {"enough": False, "returns": int(n), "days": round(span_ms / DAY_MS, 2), "volatility": None,
           "annual_return": None, "sharpe": None, "max_drawdown": None, "max_drawdown_at": None}
    if n < MIN_RETURNS or span_ms < MIN_DAYS * DAY_MS or mean is None or std is None:
        return out
    k = periods_per_year(interval_ms)
    vol, ret = std * math.sqrt(k), mean * k
    out.update(enough=True, volatility=vol, annual_return=ret, sharpe=ret / vol if vol > 1e-12 else None,
               max_drawdown=drawdown or 0.0, max_drawdown_at=drawdown_at if drawdown and drawdown < 0 else None)
    return out


def of_series(ts_ms, values, interval_ms: int, start_value: float | None = None) -> dict:
    """The statistics of values (one per candle, ts_ms the time of each). start_value: the value one candle
    before the first, the starting cash of a simulation."""
    t = np.asarray(ts_ms, dtype=np.int64)
    v = np.asarray(values, dtype=np.float64)
    if start_value is not None:
        t, v = np.concatenate(([t[0] - interval_ms], t)), np.concatenate(([start_value], v))
    if len(v) < 2:
        return finish(0, None, None, None, None, 0, interval_ms)
    r = v[1:] / v[:-1] - 1
    fall = v / np.maximum.accumulate(v) - 1
    low = int(np.argmin(fall))
    return finish(len(r), float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else None, float(fall[low]),
                  int(t[low]), float(t[-1] - t[0]), interval_ms)
