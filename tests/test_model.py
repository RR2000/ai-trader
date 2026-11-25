"""The NumPy forward pass must give the same Q-values as Keras on the same inputs."""

import os
from pathlib import Path

import numpy as np
import pytest

from conftest import MODEL
from trader.model import QNetwork

FIXTURE = Path(__file__).parent / "fixtures" / "keras_reference.npz"
MODELS = sorted(MODEL.parent.glob("*.h5"))


def test_architecture_is_the_2023_one():
    net = QNetwork(str(MODEL))
    assert net.input_shape == (30, 7)
    assert [type(layer).__name__ for layer in net.layers] == ["_LSTM", "_LSTM", "_Dense", "_Dense"]


def test_matches_tensorflow_keras_reference():
    """Inputs and outputs saved with TensorFlow 2.21 + Keras 3.15, the architecture read from the .h5 itself
    (tests/make_keras_reference.py)."""
    ref = np.load(FIXTURE)
    for name in ref["models"]:
        net = QNetwork(str(MODEL.parent / str(name)))
        ours = net.q_values(ref["inputs"])
        np.testing.assert_allclose(ours, ref[f"q_{name}"], rtol=1e-4, atol=1e-5)
        assert (ours.argmax(axis=1) == ref[f"q_{name}"].argmax(axis=1)).all()


def keras_model(path):
    os.environ.setdefault("KERAS_BACKEND", "jax")
    keras = pytest.importorskip("keras")
    model = keras.Sequential([
        keras.Input(shape=(30, 7)),
        keras.layers.LSTM(64, return_sequences=True, activation="tanh"),
        keras.layers.LSTM(32, return_sequences=False, activation="tanh"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(3, activation="linear"),
    ])
    model.load_weights(str(path))  # legacy .h5 weights, read by Keras itself
    return model


@pytest.mark.parametrize("path", MODELS, ids=[p.name for p in MODELS])
def test_matches_keras_now(path):
    """Same check against the installed Keras (CI: Keras 3 with the JAX backend)."""
    model = keras_model(path)
    rng = np.random.default_rng(7)
    x = rng.uniform(0.01, 1.01, size=(16, 30, 7)).astype(np.float32)
    expected = np.asarray(model.predict(x, verbose=0))
    ours = QNetwork(str(path)).q_values(x)
    np.testing.assert_allclose(ours, expected, rtol=1e-4, atol=1e-5)


def test_decide_is_greedy():
    net = QNetwork(str(MODEL))
    obs = np.full((30, 7), 0.5)
    action, q = net.decide(obs)
    assert action == int(np.argmax(q)) and q.shape == (3,)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        QNetwork(str(MODEL)).q_values(np.zeros((29, 7)))
