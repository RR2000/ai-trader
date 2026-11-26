"""The simulations, as Raoul defines them in simulations.toml: read and checked once, at start-up.

Any mistake (an unknown strategy, a typo in a key, a parameter out of range, two simulations with the same id)
stops the service with a message that names the simulation and the field, before anything is traded.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .market import INTERVALS
from .strategies import ConfigError, Strategy, build

ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?$")
SYMBOL = re.compile(r"^[A-Z0-9]{2,16}$")
# quote currencies recognised at the end of a Binance symbol, longest first (FDUSD before USD-anything)
QUOTES = ("FDUSD", "USDT", "USDC", "TUSD", "EUR", "TRY", "BRL", "JPY", "BTC", "ETH", "BNB")
SIM_KEYS = {"id", "name", "description", "strategy", "symbol", "interval", "start_cash", "fee", "params"}
DEFAULT_KEYS = {"symbol", "interval", "start_cash", "fee"}
DEFAULTS = {"symbol": "BTCUSDT", "interval": "1m", "start_cash": 1000.0, "fee": 0.001}
MAX_SIMULATIONS = 40


@dataclass(frozen=True)
class Simulation:
    id: str
    name: str
    description: str
    strategy: Strategy
    symbol: str
    base: str
    quote: str
    interval: str
    start_cash: float
    fee: float

    @property
    def interval_ms(self) -> int:
        return INTERVALS[self.interval]

    @property
    def fingerprint(self) -> dict:
        """What must not change once the simulation has started: a change needs a new id."""
        return {"strategy": self.strategy.kind, "params": self.strategy.params, "symbol": self.symbol,
                "interval": self.interval, "start_cash": self.start_cash, "fee": self.fee}


def split_symbol(symbol: str) -> tuple[str, str]:
    for q in QUOTES:
        if symbol.endswith(q) and len(symbol) > len(q):
            return symbol[:-len(q)], q
    raise ConfigError(f"symbol {symbol}: the quote currency must be one of {', '.join(QUOTES)}")


def _number(value, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a number")
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo:g} and {hi:g}")
    return float(value)


def _text(entry: dict, key: str, limit: int, required: bool = True) -> str:
    value = entry.get(key, "")
    if not isinstance(value, str) or (required and not value.strip()):
        raise ConfigError(f"{key} must be a non-empty string" if required else f"{key} must be a string")
    if len(value) > limit:
        raise ConfigError(f"{key} is longer than {limit} characters")
    return value.strip()


def _common(entry: dict, defaults: dict) -> dict:
    out = {}
    symbol = entry.get("symbol", defaults["symbol"])
    if not isinstance(symbol, str) or not SYMBOL.match(symbol):
        raise ConfigError("symbol must be a Binance spot symbol in capitals, like BTCUSDT")
    out["symbol"] = symbol
    interval = entry.get("interval", defaults["interval"])
    if interval not in INTERVALS:
        raise ConfigError(f"interval must be one of {', '.join(INTERVALS)}")
    out["interval"] = interval
    out["start_cash"] = _number(entry.get("start_cash", defaults["start_cash"]), "start_cash", 1, 1e9)
    out["fee"] = _number(entry.get("fee", defaults["fee"]), "fee", 0, 0.05)
    return out


def parse(data: dict, *, source: str = "simulations.toml", **context) -> list[Simulation]:
    """context: what the strategies need to set up (model_dir for the DQN)."""
    extra = sorted(set(data) - {"defaults", "simulation"})
    if extra:
        raise ConfigError(f"{source}: unknown section {', '.join(extra)} (expected [defaults] and [[simulation]])")
    raw_defaults = data.get("defaults", {})
    if not isinstance(raw_defaults, dict) or set(raw_defaults) - DEFAULT_KEYS:
        raise ConfigError(f"{source}: [defaults] may only set {', '.join(sorted(DEFAULT_KEYS))}")
    try:
        defaults = _common(raw_defaults, DEFAULTS)
    except ConfigError as e:
        raise ConfigError(f"{source}: [defaults]: {e}") from None
    entries = data.get("simulation")
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{source}: no simulation defined (add a [[simulation]] entry)")
    if len(entries) > MAX_SIMULATIONS:
        raise ConfigError(f"{source}: {len(entries)} simulations, at most {MAX_SIMULATIONS}")
    sims, seen = [], set()
    for n, entry in enumerate(entries, 1):
        label = f"simulation #{n}"
        try:
            if not isinstance(entry, dict):
                raise ConfigError("must be a table")
            sim_id = entry.get("id")
            if not isinstance(sim_id, str) or not ID.match(sim_id):
                raise ConfigError("id must be 1-40 lowercase letters, digits or dashes, like rsi-14")
            label = f"simulation {sim_id!r}"
            if sim_id in seen:
                raise ConfigError("the same id is used twice")
            seen.add(sim_id)
            unknown = sorted(set(entry) - SIM_KEYS)
            if unknown:
                raise ConfigError(f"unknown key {', '.join(unknown)} (known: {', '.join(sorted(SIM_KEYS))})")
            name = _text(entry, "name", 60)
            description = _text(entry, "description", 300, required=False)
            common = _common(entry, defaults)
            base, quote = split_symbol(common["symbol"])
            kind = entry.get("strategy")
            if not isinstance(kind, str):
                raise ConfigError("strategy is missing")
            strategy = build(kind, entry.get("params", {}), **context)
        except ConfigError as e:
            raise ConfigError(f"{source}: {label}: {e}") from None
        sims.append(Simulation(sim_id, name, description, strategy, common["symbol"], base, quote, common["interval"],
                               common["start_cash"], common["fee"]))
    return sims


def load(path: Path | str, **context) -> list[Simulation]:
    path = Path(path)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found (set SIMULATIONS to its path)") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path.name}: not valid TOML: {e}") from None
    return parse(data, source=path.name, **context)
