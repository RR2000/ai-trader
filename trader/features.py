"""The observation the 2023 model was trained on, rebuilt without pandas, ta and scikit-learn.

ModelTrainer.py took one-minute candles and kept Close and Volume, added five indicators from the
`ta` library (SMA 10, SMA 30, RSI 14, MACD 12/26 and its 9 signal), dropped the rows with gaps,
min-max scaled every column to [0.01, 1.01] and fed the network windows of 30 consecutive rows.
The functions below reproduce those formulas (pandas ewm with adjust=False, ta's RSI and MACD,
sklearn's MinMaxScaler); tests/test_features.py checks them against the original libraries.

One deliberate difference: in training the scaler was fit on the whole 107-row episode
(30 + 7 * 11 rows, ModelTrainer.py), so each window was scaled knowing the future. Live there is
no future: the scaler is fit on the last 107 rows, the same length, ending at the latest candle.
"""

from __future__ import annotations

import numpy as np

FEATURES = ("Close", "Volume", "SMA_10", "SMA_30", "RSI", "MACD", "MACD_signal")
WINDOW = 30  # rows per observation
SCALE_ROWS = 30 + 7 * 11  # rows the scaler sees: the size of a training episode
FEATURE_RANGE = (0.01, 1.01)
WARMUP = 33  # first row with every indicator defined (MACD signal: 26 + 9 - 2)


def sma(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= window:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def ewm(x: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    """pandas Series.ewm(alpha=..., adjust=False, min_periods=...).mean() for leading-NaN inputs."""
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) == 0:
        return out
    start = valid[0]
    y = x[start]
    for t in range(start, len(x)):
        if t > start:
            y = (1.0 - alpha) * y + alpha * x[t]
        if t - start + 1 >= min_periods:
            out[t] = y
    return out


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """ta's _ema: ewm(span=n, min_periods=n, adjust=False)."""
    return ewm(x, 2.0 / (span + 1.0), span)


def rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    diff = np.diff(close, prepend=np.nan)
    up = np.where(diff > 0, diff, 0.0)  # the first (NaN) difference becomes 0, as in ta
    down = np.where(diff < 0, -diff, 0.0)
    emaup = ewm(up, 1.0 / window, window)
    emadn = ewm(down, 1.0 / window, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + emaup / emadn)
    return np.where(emadn == 0, 100.0, out)


def indicators(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Rows of FEATURES, starting from the first row where every indicator is defined."""
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    macd = ema(close, 12) - ema(close, 26)
    table = np.column_stack([close, volume, sma(close, 10), sma(close, 30), rsi(close), macd, ema(macd, 9)])
    return table[~np.isnan(table).any(axis=1)]


def min_max_scale(rows: np.ndarray) -> np.ndarray:
    """sklearn MinMaxScaler(feature_range=(0.01, 1.01)).fit_transform(rows)."""
    lo, hi = FEATURE_RANGE
    data_min = rows.min(axis=0)
    data_range = rows.max(axis=0) - data_min
    data_range[data_range < 10 * np.finfo(np.float64).eps] = 1.0  # sklearn's _handle_zeros_in_scale
    scale = (hi - lo) / data_range
    return rows * scale + (lo - data_min * scale)


def observation(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """The (WINDOW, len(FEATURES)) window ending at the latest candle."""
    table = indicators(close, volume)
    if len(table) < SCALE_ROWS:
        raise ValueError(f"need at least {SCALE_ROWS + WARMUP} candles, got {len(close)}")
    scaled = min_max_scale(table[-SCALE_ROWS:])
    return scaled[-WINDOW:]
