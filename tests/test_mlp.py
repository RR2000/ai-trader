"""The 2026 network: features without look-ahead, the same live as in training, NumPy inference = scikit-learn."""

import numpy as np
import pytest

from conftest import MODEL
from trader import mlfeatures
from trader.market import Candles
from trader.strategies import ConfigError, Position, build

MLP_DIR = MODEL.parent.parent / "mlp"


def series(n=1200, seed=4):
    rng = np.random.default_rng(seed)
    return 60_000 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), rng.gamma(2.0, 400.0, n)


def test_features_are_what_they_say():
    close, volume = series()
    f = mlfeatures.table(close, volume)
    assert f.shape == (len(close), len(mlfeatures.NAMES))
    assert np.isnan(f[:mlfeatures.WARMUP]).all() and not np.isnan(f[mlfeatures.WARMUP:]).any()
    t = 500
    col = dict(zip(mlfeatures.NAMES, f[t]))
    assert col["ret_24"] == pytest.approx(np.log(close[t] / close[t - 24]))
    assert col["sma_168"] == pytest.approx(np.log(close[t] / close[t - 167:t + 1].mean()))
    r1 = np.diff(np.log(close))
    assert col["vol_24"] == pytest.approx(r1[t - 24:t].std(), rel=1e-6)
    lv = np.log1p(volume[t - 167:t + 1])
    assert col["volume_z"] == pytest.approx((lv[-1] - lv.mean()) / lv.std(), rel=1e-6)


def test_no_look_ahead_and_the_same_live_as_in_training():
    close, volume = series()
    full = mlfeatures.table(close, volume)
    changed = close.copy()
    changed[800:] *= 1.5  # a different future
    assert np.array_equal(mlfeatures.table(changed, volume)[:800], full[:800], equal_nan=True)
    live = mlfeatures.table(close[-400:], volume[-400:])  # what the service sees: the latest 400 candles
    np.testing.assert_allclose(live[-1], full[-1], rtol=1e-9, atol=1e-12)


def test_numpy_inference_matches_scikit_learn(tmp_path):
    neural = pytest.importorskip("sklearn.neural_network")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 12))
    y = (x[:, 0] + 0.5 * x[:, 3] > 0).astype(int)
    clf = neural.MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=300, random_state=0).fit(x, y)
    m = mlfeatures.MLP(np.zeros(12), np.ones(12), tuple(clf.coefs_), tuple(clf.intercepts_),
                       {"layers": 3, "horizon_hours": 24, "features": list(mlfeatures.NAMES), "hidden": [8, 4]})
    m.save(tmp_path / "m.npz")
    again = mlfeatures.MLP.load(tmp_path / "m.npz")
    np.testing.assert_allclose(again.proba(x), clf.predict_proba(x)[:, 1], atol=1e-12)
    assert again.meta["horizon_hours"] == 24


def test_policy_thresholds():
    assert [mlfeatures.action(p, 0.55, 0.45) for p in (0.6, 0.56, 0.5, 0.45, 0.3)] == [0, 0, 2, 2, 1]


def test_the_shipped_model_trades():
    s = build("mlp", {}, model_dir=MLP_DIR)
    assert s.lookback == 400 and s.horizon == 24
    assert s.net.meta["features"] == list(mlfeatures.NAMES)
    close, volume = series(400)
    c = Candles(list(range(0, 400 * 3_600_000, 3_600_000)), list(close), list(volume), 3_600_000)
    d = s.decide(c, Position(1000.0, 0.0))
    assert d.action in (0, 1, 2) and "chance of a higher price in 24 h" in d.reason
    p = float(s.net.proba(mlfeatures.table(close, volume)[-1])[0])
    assert d.action == mlfeatures.action(p, 0.55, 0.45)
    with pytest.raises(ValueError):
        s.decide(Candles([0] * 100, list(close[:100]), list(volume[:100]), 3_600_000), Position(1000.0, 0.0))
    with pytest.raises(ConfigError, match="not found"):
        build("mlp", {"model": "nope.npz"}, model_dir=MLP_DIR)
    with pytest.raises(ConfigError, match="between"):
        build("mlp", {"enter": 0.4}, model_dir=MLP_DIR)
