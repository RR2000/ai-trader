"""simulations.toml: the shipped file is valid, and every mistake stops the start with a clear message."""

import tomllib
from pathlib import Path

import pytest

from conftest import MODEL, simulations
from trader.config import load
from trader.strategies import ConfigError

ROOT = Path(__file__).resolve().parent.parent
GOOD = """
[[simulation]]
id = "rsi-eth"
name = "RSI on ETH"
strategy = "rsi"
symbol = "ETHUSDT"
interval = "15m"
"""


def test_the_shipped_file_is_valid():
    sims = load(ROOT / "simulations.toml", model_dir=MODEL.parent, networks={})
    ids = [s.id for s in sims]
    assert ids[0] == "dqn-2023" and len(ids) == len(set(ids)) >= 5
    assert {s.strategy.kind for s in sims} >= {"dqn", "sma-cross", "rsi", "bollinger", "dca", "buy-and-hold"}
    dqn = sims[0]
    # exactly the configuration the migration gives to the history of the single simulation
    assert dqn.fingerprint == {"strategy": "dqn", "params": {"model": "trading_model_episode_30.h5"},
                               "symbol": "BTCUSDT", "interval": "1m", "start_cash": 1000.0, "fee": 0.001}
    assert all(s.start_cash == 1000.0 and s.fee == 0.001 and s.symbol == "BTCUSDT" for s in sims)


def test_defaults_and_pairs():
    (s,) = simulations("[defaults]\nstart_cash = 500\nfee = 0.00075\n" + GOOD)
    assert (s.symbol, s.base, s.quote, s.interval, s.start_cash, s.fee) == ("ETHUSDT", "ETH", "USDT", "15m", 500.0, 0.00075)
    assert s.strategy.params == {"period": 14, "oversold": 30.0, "overbought": 70.0}
    assert s.description == ""
    (s,) = simulations(GOOD.replace("ETHUSDT", "SOLBTC"))
    assert (s.base, s.quote) == ("SOL", "BTC")


@pytest.mark.parametrize("text,message", [
    ("", "no simulation defined"),
    (GOOD + GOOD, "simulation 'rsi-eth': the same id is used twice"),
    (GOOD.replace('"rsi-eth"', '"RSI eth"'), "simulation #1: id must be"),
    (GOOD.replace('name = "RSI on ETH"', 'nmae = "x"'), "unknown key nmae"),
    (GOOD.replace('name = "RSI on ETH"\n', ""), "name must be a non-empty string"),
    (GOOD.replace('"rsi"', '"rsii"'), "unknown strategy 'rsii'"),
    (GOOD.replace('strategy = "rsi"\n', ""), "strategy is missing"),
    (GOOD.replace('"15m"', '"10m"'), "interval must be one of"),
    (GOOD.replace('"ETHUSDT"', '"eth-usdt"'), "symbol must be"),
    (GOOD.replace('"ETHUSDT"', '"ETHXYZ"'), "quote currency"),
    (GOOD + "start_cash = 0\n", "start_cash must be between"),
    (GOOD + "fee = 0.5\n", "fee must be between"),
    (GOOD + "fee = true\n", "fee must be a number"),
    (GOOD + "[simulation.params]\nperiod = 0\n", "simulation 'rsi-eth': params.period must be between"),
    ("[defaults]\ninterval = \"2m\"\n" + GOOD, r"\[defaults\]: interval"),
    ("[defaults]\nname = \"x\"\n" + GOOD, r"\[defaults\] may only set"),
    ("[simulations]\n" + GOOD, "unknown section simulations"),
    ('[[simulation]]\nid = "d"\nname = "D"\nstrategy = "dqn"\n[simulation.params]\nmodel = "nope.h5"\n', "model file nope.h5 not found"),
])
def test_mistakes_are_explained(text, message):
    with pytest.raises(ConfigError, match=message):
        simulations(text)


def test_bad_toml_and_missing_file(tmp_path):
    bad = tmp_path / "simulations.toml"
    bad.write_text("[[simulation]\nid = 1", encoding="utf-8")
    with pytest.raises(ConfigError, match="simulations.toml: not valid TOML"):
        load(bad)
    with pytest.raises(ConfigError, match="file not found"):
        load(tmp_path / "nope.toml")


def test_the_readme_lists_every_strategy_and_parameter():
    from trader.strategies import STRATEGIES

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for kind, cls in STRATEGIES.items():
        assert f"`{kind}`" in readme
        for name in cls.PARAMS:
            assert f"`{name}`" in readme, (kind, name)
    assert tomllib.loads((ROOT / "simulations.toml").read_text(encoding="utf-8"))
