"""Inference of the 2023 Keras model in plain NumPy.

The network trained in 2023 (DQN.py) is a Sequential model:
LSTM(64, return_sequences) -> LSTM(32) -> Dense(16, relu) -> Dense(3, linear).
Its .h5 file holds the architecture (model_config) and the weights, so h5py is enough to
read it and the forward pass is a few lines of NumPy: no TensorFlow in the image.
tests/test_model.py checks that the outputs match Keras on the same inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import h5py
import numpy as np

ACTIONS = ("buy", "sell", "hold")  # index = network output, as in TradingEnvironment.py


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # stable for large |x|: 1 / (1 + e^-x) without overflow warnings
    return np.exp(-np.logaddexp(0.0, -x))


@dataclass(frozen=True)
class _LSTM:
    kernel: np.ndarray  # (input_dim, 4 * units), gate order i, f, c, o (Keras)
    recurrent: np.ndarray  # (units, 4 * units)
    bias: np.ndarray  # (4 * units,)
    return_sequences: bool

    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch, steps, _ = x.shape
        units = self.recurrent.shape[0]
        h = np.zeros((batch, units))
        c = np.zeros((batch, units))
        xw = x @ self.kernel + self.bias  # input part of every step at once
        out = np.empty((batch, steps, units)) if self.return_sequences else None
        for t in range(steps):
            z = xw[:, t, :] + h @ self.recurrent
            i = _sigmoid(z[:, :units])
            f = _sigmoid(z[:, units:2 * units])
            g = np.tanh(z[:, 2 * units:3 * units])
            o = _sigmoid(z[:, 3 * units:])
            c = f * c + i * g
            h = o * np.tanh(c)
            if out is not None:
                out[:, t, :] = h
        return out if out is not None else h


@dataclass(frozen=True)
class _Dense:
    kernel: np.ndarray
    bias: np.ndarray
    relu: bool

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x @ self.kernel + self.bias
        return np.maximum(y, 0.0) if self.relu else y


class QNetwork:
    """The trained Q-network: (window, features) in, one Q-value per action out."""

    def __init__(self, path: str):
        with h5py.File(path, "r") as f:
            config = json.loads(f.attrs["model_config"])
            weights = f["model_weights"]
            self.layers = []
            self.input_shape = None
            for layer in config["config"]["layers"]:
                kind, cfg = layer["class_name"], layer["config"]
                if kind == "InputLayer":
                    self.input_shape = tuple(cfg["batch_input_shape"][1:])
                    continue
                group = weights[cfg["name"]]
                names = [n.decode() if isinstance(n, bytes) else n for n in group.attrs["weight_names"]]
                arrays = {n.rsplit("/", 1)[-1].split(":")[0]: np.asarray(group[n], dtype=np.float64) for n in names}
                if kind == "LSTM":
                    expected = {"activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": True,
                                "go_backwards": False, "stateful": False}
                    wrong = {k: cfg.get(k) for k, v in expected.items() if cfg.get(k) != v}
                    if wrong:
                        raise ValueError(f"unsupported LSTM settings in {cfg['name']}: {wrong}")
                    self.layers.append(_LSTM(arrays["kernel"], arrays["recurrent_kernel"], arrays["bias"],
                                             bool(cfg["return_sequences"])))
                elif kind == "Dense":
                    if cfg["activation"] not in ("relu", "linear") or not cfg["use_bias"]:
                        raise ValueError(f"unsupported Dense settings in {cfg['name']}")
                    self.layers.append(_Dense(arrays["kernel"], arrays["bias"], cfg["activation"] == "relu"))
                else:
                    raise ValueError(f"unsupported layer {kind}")
        if self.input_shape is None:
            raise ValueError("model without an input layer")

    def q_values(self, observations: np.ndarray) -> np.ndarray:
        """observations: (batch, window, features) or (window, features). Returns (batch, actions)."""
        x = np.asarray(observations, dtype=np.float64)
        if x.ndim == 2:
            x = x[np.newaxis]
        if x.shape[1:] != self.input_shape:
            raise ValueError(f"expected observations of shape {self.input_shape}, got {x.shape[1:]}")
        for layer in self.layers:
            x = layer(x)
        return x

    def decide(self, observation: np.ndarray) -> tuple[int, np.ndarray]:
        """Greedy action (the trained policy, no exploration) and its Q-values."""
        q = self.q_values(observation)[0]
        return int(np.argmax(q)), q
