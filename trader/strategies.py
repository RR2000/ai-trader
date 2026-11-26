"""The strategies a simulation can run: the 2023 deep Q-network and a few classic rules.

A strategy looks at the latest closed candles (exactly `lookback` of them) and at the portfolio, and returns
a Decision: buy (all the cash, or a fixed amount), sell everything, or hold. It keeps no state of its own:
everything it needs is in the candles and in the position stored in the database, so a restart changes nothing.
The accounting (fees, balances, buy and hold) is the same for all of them, in portfolio.py and store.py.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import features
from .market import Candles, MAX_LIMIT
from .portfolio import BUY, HOLD, SELL

ACTIONS = ("buy", "sell", "hold")  # index = action code


class ConfigError(ValueError):
    """A simulation definition that cannot run: the message says which one and why."""


@dataclass(frozen=True)
class Position:
    cash: float  # quote currency left
    crypto: float  # base asset held
    last_buy_at: datetime | None = None  # candle of the latest buy, None before the first one


@dataclass(frozen=True)
class Decision:
    action: int  # BUY, SELL or HOLD
    reason: str  # one line for the page: what the strategy saw
    amount: float | None = None  # size of a buy in the quote currency; None = all the cash
    q: tuple[float, float, float] | None = None  # the network's Q-values (DQN only)


@dataclass(frozen=True)
class Param:
    kind: type  # int, float or str
    default: object
    help: str
    lo: float | None = None
    hi: float | None = None


def find_model(model_dir, name: str) -> Path:
    """A model file by name in MODEL_DIR: one directory, or several separated by os.pathsep (or a list)."""
    dirs = model_dir if isinstance(model_dir, (list, tuple)) else str(model_dir or ".").split(os.pathsep)
    if Path(name).name == name:
        for d in dirs:
            if (Path(d) / name).is_file():
                return Path(d) / name
    raise ConfigError(f"model file {name} not found in {os.pathsep.join(str(d) for d in dirs)}")


def _fmt(x: float) -> str:
    return f"{x:,.2f}"


DURATION = re.compile(r"^([1-9][0-9]{0,3})([mhd])$")


def duration_words(text: str) -> str:
    """"1d" -> "day", "12h" -> "12 hours"."""
    n, unit = int(text[:-1]), {"m": "minute", "h": "hour", "d": "day"}[text[-1]]
    return unit if n == 1 else f"{n} {unit}s"


def parse_duration(text: str) -> timedelta:
    m = DURATION.match(text) if isinstance(text, str) else None
    if not m:
        raise ValueError("a duration like 30m, 12h or 7d")
    n, unit = int(m.group(1)), m.group(2)
    return timedelta(minutes=n) if unit == "m" else timedelta(hours=n) if unit == "h" else timedelta(days=n)


class Strategy:
    kind = ""  # the name used in simulations.toml
    title = ""  # for the page
    PARAMS: dict[str, Param] = {}

    def __init__(self, params: dict, **context):
        self.params = self._check(params)
        self.setup(**context)

    @classmethod
    def _check(cls, params: dict) -> dict:
        if not isinstance(params, dict):
            raise ConfigError("params must be a table")
        unknown = sorted(set(params) - set(cls.PARAMS))
        if unknown:
            known = ", ".join(cls.PARAMS) or "none"
            raise ConfigError(f"unknown parameter {', '.join(unknown)} for strategy {cls.kind} (known: {known})")
        out = {}
        for name, p in cls.PARAMS.items():
            value = params.get(name, p.default)
            if p.kind is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, p.kind) or isinstance(value, bool):
                raise ConfigError(f"params.{name} must be {'an integer' if p.kind is int else 'a number' if p.kind is float else 'a string'}")
            if p.kind in (int, float) and (not math.isfinite(value) or (p.lo is not None and value < p.lo)
                                           or (p.hi is not None and value > p.hi)):
                raise ConfigError(f"params.{name} must be between {p.lo:g} and {p.hi:g}")
            out[name] = value
        return out

    def setup(self, **context):
        """Extra checks and set-up once the parameters are known."""

    @property
    def lookback(self) -> int:
        """How many closed candles decide() needs."""
        return 1

    def decide(self, candles: Candles, position: Position) -> Decision:
        raise NotImplementedError

    def explain(self, quote: str = "USDT") -> str:
        """How it decides, in plain words, with its parameters (quote: the cash currency)."""
        raise NotImplementedError


class DQN2023(Strategy):
    kind = "dqn"
    title = "Deep Q-network (2023)"
    PARAMS = {"model": Param(str, "trading_model_episode_30.h5", "the .h5 file of the trained network, in MODEL_DIR")}

    def setup(self, model_dir: Path | str | None = None, networks: dict | None = None, **_):
        from .model import QNetwork

        path = find_model(model_dir, self.params["model"])
        cache = networks if networks is not None else {}
        if str(path) not in cache:  # one copy of the weights however many simulations use it
            cache[str(path)] = QNetwork(str(path))
        self.net = cache[str(path)]

    @property
    def lookback(self) -> int:
        # the closed candles of a 500-candle request, as since the first live day: the indicators' warm-up
        # (EMA, RSI) depends on where the history starts, so this stays exactly as it was
        return 499

    def decide(self, candles, position):
        obs = features.observation(np.asarray(candles.close), np.asarray(candles.volume))
        action, q = self.net.decide(obs)
        q = tuple(float(v) for v in q)
        return Decision(action, f"Q-values: buy {q[0]:.4f}, sell {q[1]:.4f}, hold {q[2]:.4f}: {ACTIONS[action]}", q=q)

    def explain(self, quote="USDT"):
        return ("The deep Q-network trained in April 2023 (two LSTM layers, then two dense layers). Every minute it "
                "gets the last 30 one-minute candles as price, volume, SMA 10, SMA 30, RSI 14, MACD and MACD signal, "
                "min-max scaled, and picks the action with the highest Q-value: buy with all the cash, sell everything, "
                "or hold. It was trained on a very small slice of data, so do not expect it to be clever.")


class Mlp(Strategy):
    kind = "mlp"
    title = "Small neural network (2026)"
    PARAMS = {"model": Param(str, "mlp_btcusdt_1h_2026-10.npz", "the .npz file written by training/train_mlp.py, in MODEL_DIR"),
              "enter": Param(float, 0.55, "buy when the predicted chance of a rise is above this", 0.5, 1),
              "leave": Param(float, 0.45, "sell when it is below this", 0, 0.5)}

    def setup(self, model_dir: Path | str | None = None, **_):
        from .mlfeatures import MLP

        self.net = MLP.load(find_model(model_dir, self.params["model"]))
        self.horizon = int(self.net.meta["horizon_hours"])

    @property
    def lookback(self):
        # 169 candles define every feature; the RSI's smoothing forgets where it started after a few hundred
        return 400

    def decide(self, candles, position):
        from . import mlfeatures

        rows = mlfeatures.table(candles.close, candles.volume)
        if np.isnan(rows[-1]).any():
            raise ValueError(f"need at least {mlfeatures.WARMUP + 1} candles, got {len(candles.close)}")
        p = float(self.net.proba(rows[-1])[0])
        action = mlfeatures.action(p, self.params["enter"], self.params["leave"])
        return Decision(action, f"chance of a higher price in {self.horizon} h: {p:.1%}: {ACTIONS[action]}")

    def explain(self, quote="USDT"):
        p, m = self.params, self.net.meta
        return (f"A small neural network ({len(m['features'])} inputs, {m['hidden'][0]} hidden units) trained in "
                f"2026 on hourly BTC/USDT candles from January 2019 to {m['trained_until']}. From the last week of "
                f"hourly closes and volumes (returns over 1 to 168 hours, volatility, RSI, distance from two moving "
                f"averages, volume) it estimates the chance that the price is higher {self.horizon} hours later. "
                f"Above {p['enter']:.0%} it buys with all the cash, below {p['leave']:.0%} it sells everything, in "
                f"between it holds. Its walk-forward test, where it never saw the months it traded, is in the "
                f"README, losses included.")


class SmaCross(Strategy):
    kind = "sma-cross"
    title = "Moving-average crossover"
    PARAMS = {"fast": Param(int, 20, "candles of the fast simple moving average", 2, 400),
              "slow": Param(int, 50, "candles of the slow simple moving average", 3, MAX_LIMIT - 1)}

    def setup(self, **_):
        if self.params["fast"] >= self.params["slow"]:
            raise ConfigError("params.fast must be smaller than params.slow")

    @property
    def lookback(self):
        return self.params["slow"]

    def decide(self, candles, position):
        close = np.asarray(candles.close)
        f, s = self.params["fast"], self.params["slow"]
        if len(close) < s:
            raise ValueError(f"need {s} candles, got {len(close)}")
        fast, slow = close[-f:].mean(), close[-s:].mean()
        action = BUY if fast > slow else SELL if fast < slow else HOLD
        word = "above" if fast > slow else "below" if fast < slow else "equal to"
        return Decision(action, f"SMA {f} {_fmt(fast)} {word} SMA {s} {_fmt(slow)}: {ACTIONS[action]}")

    def explain(self, quote="USDT"):
        f, s = self.params["fast"], self.params["slow"]
        return (f"Trend following. At every candle it compares the average close of the last {f} candles with the "
                f"average of the last {s}. While the fast average is above the slow one it holds the coin (it buys "
                f"with all the cash when the fast one crosses above); when the fast one drops below, it sells everything.")


class Rsi(Strategy):
    kind = "rsi"
    title = "RSI mean reversion"
    PARAMS = {"period": Param(int, 14, "candles of the relative strength index (Wilder smoothing)", 2, 49),
              "oversold": Param(float, 30.0, "buy when the RSI is below this", 1, 99),
              "overbought": Param(float, 70.0, "sell when the RSI is above this", 1, 99)}

    def setup(self, **_):
        if self.params["oversold"] >= self.params["overbought"]:
            raise ConfigError("params.oversold must be smaller than params.overbought")

    @property
    def lookback(self):
        return self.params["period"] * 20  # enough history for the smoothing to forget where it started

    def decide(self, candles, position):
        close = np.asarray(candles.close, dtype=np.float64)
        p = self.params
        if len(close) < p["period"] + 1:
            raise ValueError(f"need {p['period'] + 1} candles, got {len(close)}")
        value = float(features.rsi(close, p["period"])[-1])
        action = BUY if value < p["oversold"] else SELL if value > p["overbought"] else HOLD
        return Decision(action, f"RSI {p['period']} at {value:.1f} (buy below {p['oversold']:g}, "
                                f"sell above {p['overbought']:g}): {ACTIONS[action]}")

    def explain(self, quote="USDT"):
        p = self.params
        return (f"Mean reversion. The relative strength index over {p['period']} candles measures how strong the "
                f"recent rises were compared with the falls, from 0 to 100. Below {p['oversold']:g} the coin is "
                f"considered oversold and it buys with all the cash; above {p['overbought']:g} it is considered "
                f"overbought and it sells everything. In between it holds.")


class Bollinger(Strategy):
    kind = "bollinger"
    title = "Bollinger Bands"
    PARAMS = {"period": Param(int, 20, "candles of the moving average", 2, MAX_LIMIT - 1),
              "width": Param(float, 2.0, "distance of the bands, in standard deviations", 0.1, 10)}

    @property
    def lookback(self):
        return self.params["period"]

    def decide(self, candles, position):
        p = self.params
        close = np.asarray(candles.close[-p["period"]:], dtype=np.float64)
        if len(close) < p["period"]:
            raise ValueError(f"need {p['period']} candles, got {len(close)}")
        mid, sd = close.mean(), close.std()
        lower, upper, last = mid - p["width"] * sd, mid + p["width"] * sd, close[-1]
        action = BUY if last < lower else SELL if last > upper else HOLD
        where = "below the lower band" if last < lower else "above the upper band" if last > upper else "inside the bands"
        return Decision(action, f"close {_fmt(last)} {where} ({_fmt(lower)} to {_fmt(upper)}): {ACTIONS[action]}")

    def explain(self, quote="USDT"):
        p = self.params
        return (f"Mean reversion. The bands sit {p['width']:g} standard deviations above and below the average close "
                f"of the last {p['period']} candles. A close below the lower band buys with all the cash, a close "
                f"above the upper band sells everything; inside the bands it holds.")


class Dca(Strategy):
    kind = "dca"
    title = "Dollar-cost averaging"
    PARAMS = {"amount": Param(float, 25.0, "quote currency spent at every buy", 0.01, 1e9),
              "every": Param(str, "1d", "time between buys, like 12h or 7d")}

    def setup(self, **_):
        try:
            self.every = parse_duration(self.params["every"])
        except ValueError as e:
            raise ConfigError(f"params.every must be {e}") from None

    def decide(self, candles, position):
        ts = datetime.fromtimestamp(candles.last_close_time_ms / 1000, tz=timezone.utc)
        due = position.last_buy_at is None or ts - position.last_buy_at >= self.every
        amount = self.params["amount"]
        if position.cash <= 0:
            return Decision(HOLD, "all the cash is spent: it only holds now")
        if due:
            return Decision(BUY, f"scheduled buy of {_fmt(min(amount, position.cash))}, one every "
                                 f"{duration_words(self.params['every'])}", amount=amount)
        nxt = position.last_buy_at + self.every
        return Decision(HOLD, f"next buy at {nxt.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")

    def explain(self, quote="USDT"):
        p = self.params
        return (f"No prediction at all: it buys {_fmt(p['amount'])} {quote} worth of the coin every "
                f"{duration_words(p['every'])}, "
                f"whatever the price, until the cash runs out, and never sells. The cash waiting to be spent is "
                f"counted in the value.")


class BuyAndHold(Strategy):
    kind = "buy-and-hold"
    title = "Buy and hold"

    def decide(self, candles, position):
        if position.cash > 0:
            return Decision(BUY, "first candle: buys with all the cash")
        return Decision(HOLD, "bought at the start, never sells")

    def explain(self, quote="USDT"):
        return ("The reference: it buys with all the cash at the first candle, paying one fee, and never sells. "
                "Every other simulation is compared with its own buy and hold from the moment it started.")


STRATEGIES: dict[str, type[Strategy]] = {cls.kind: cls for cls in (DQN2023, Mlp, SmaCross, Rsi, Bollinger, Dca, BuyAndHold)}


def build(kind: str, params: dict, **context) -> Strategy:
    cls = STRATEGIES.get(kind)
    if cls is None:
        raise ConfigError(f"unknown strategy {kind!r} (known: {', '.join(STRATEGIES)})")
    s = cls(params, **context)
    if s.lookback + 1 > MAX_LIMIT:
        raise ConfigError(f"needs {s.lookback} candles, at most {MAX_LIMIT - 1} can be read at once")
    return s

