"""Features and inference of the small network trained by training/train_mlp.py, in plain NumPy.

The same code builds the features for training (the whole history at once) and live (the latest candles),
so the model sees in production exactly what it was trained on. Every feature of row t uses only the
candles up to t: closes and volumes, nothing from the future.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .features import rsi

RETURNS = (1, 4, 12, 24, 72, 168)  # log return over the last k candles
NAMES = tuple(f"ret_{k}" for k in RETURNS) + ("vol_24", "vol_168", "rsi_14", "sma_24", "sma_168", "volume_z")
WARMUP = 169  # first row with every feature defined (168 candles of history before it)


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    m = _rolling_mean(x, n)
    m2 = _rolling_mean(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def table(close, volume) -> np.ndarray:
    """One row of NAMES per candle; the first WARMUP rows are NaN."""
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    logc = np.log(close)
    cols = []
    for k in RETURNS:
        r = np.full(len(close), np.nan)
        r[k:] = logc[k:] - logc[:-k]
        cols.append(r)
    r1 = cols[0]
    r1z = np.where(np.isnan(r1), 0.0, r1)
    v24, v168 = _rolling_std(r1z, 24), _rolling_std(r1z, 168)
    v24[:24] = np.nan
    v168[:168] = np.nan
    cols += [v24, v168]
    cols.append(rsi(close, 14) / 100.0 - 0.5)
    cols.append(logc - np.log(_rolling_mean(close, 24)))
    cols.append(logc - np.log(_rolling_mean(close, 168)))
    lv = np.log1p(volume)
    mu, sd = _rolling_mean(lv, 168), _rolling_std(lv, 168)
    cols.append((lv - mu) / np.where(sd > 1e-12, sd, 1.0))
    out = np.column_stack(cols)
    out[:WARMUP] = np.nan
    return out


@dataclass(frozen=True)
class MLP:
    """standardise -> Dense(relu) ... -> Dense(1, sigmoid): the probability that the price is higher in
    `horizon` candles. Weights and metadata in one .npz file."""
    mean: np.ndarray
    std: np.ndarray
    weights: tuple
    biases: tuple
    meta: dict

    @classmethod
    def load(cls, path) -> "MLP":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            n = meta["layers"]
            return cls(f["mean"], f["std"], tuple(f[f"w{i}"] for i in range(n)), tuple(f[f"b{i}"] for i in range(n)),
                       meta)

    def save(self, path):
        arrays = {"mean": self.mean, "std": self.std, "meta": np.array(json.dumps(self.meta, sort_keys=True))}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"], arrays[f"b{i}"] = w, b
        np.savez(path, **arrays)

    def proba(self, rows: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(rows) - self.mean) / self.std
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return np.exp(-np.logaddexp(0.0, -x[:, 0]))  # sigmoid without overflow


def action(p: float, enter: float, leave: float) -> int:
    """The policy on top of the probability: buy above `enter`, sell below `leave`, hold in between
    (the gap keeps it from paying fees on every small wobble). 0 buy, 1 sell, 2 hold."""
    return 0 if p > enter else 1 if p < leave else 2
