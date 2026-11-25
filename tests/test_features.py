"""The observation must be the one ModelTrainer.py built with pandas, ta and scikit-learn."""

import numpy as np
import pytest

from trader import features

pd = pytest.importorskip("pandas")
ta = pytest.importorskip("ta")
preprocessing = pytest.importorskip("sklearn.preprocessing")


def candles(n=500, seed=1):
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    volume = rng.gamma(2.0, 4.0, n)
    return close, volume


def trainer_indicators(close, volume):
    """ModelTrainer.get_data_and_technical_indicators, verbatim apart from the CSV."""
    tmp = pd.DataFrame({"Close": close, "Volume": volume})
    tmp["SMA_10"] = ta.trend.sma_indicator(tmp["Close"], window=10)
    tmp["SMA_30"] = ta.trend.sma_indicator(tmp["Close"], window=30)
    tmp["RSI"] = ta.momentum.RSIIndicator(tmp["Close"]).rsi()
    macd_indicator = ta.trend.MACD(tmp["Close"])
    tmp["MACD"] = macd_indicator.macd()
    tmp["MACD_signal"] = macd_indicator.macd_signal()
    tmp.dropna(inplace=True)
    return tmp


def test_indicators_match_ta():
    close, volume = candles()
    ours = features.indicators(close, volume)
    theirs = trainer_indicators(close, volume)
    assert list(theirs.columns) == list(features.FEATURES)
    assert len(ours) == len(theirs) == len(close) - features.WARMUP
    np.testing.assert_allclose(ours, theirs.to_numpy(), rtol=1e-10, atol=1e-8)


def test_rsi_flat_and_rising_prices():
    rising = np.linspace(100, 200, 80)
    rows = features.indicators(rising, np.ones(80))
    assert np.allclose(rows[:, 4], 100.0)  # no losses: RSI 100, as in ta


def test_scaler_matches_sklearn():
    close, volume = candles()
    rows = features.indicators(close, volume)[-features.SCALE_ROWS:]
    rows[:, 1] = 3.0  # a constant column: sklearn maps it to the low end of the range
    sk = preprocessing.MinMaxScaler(feature_range=(0.01, 1.01)).fit_transform(rows)
    np.testing.assert_allclose(features.min_max_scale(rows), sk, rtol=1e-12, atol=1e-12)


def test_observation_is_the_last_window_of_the_scaled_episode():
    close, volume = candles()
    table = trainer_indicators(close, volume).tail(features.SCALE_ROWS)
    scaled = preprocessing.MinMaxScaler(feature_range=(0.01, 1.01)).fit_transform(table)
    obs = features.observation(close, volume)
    assert obs.shape == (30, 7)
    np.testing.assert_allclose(obs, scaled[-30:], rtol=1e-9, atol=1e-9)
    assert obs.min() >= 0.01 - 1e-12 and obs.max() <= 1.01 + 1e-12


def test_observation_needs_enough_candles():
    close, volume = candles(features.SCALE_ROWS + features.WARMUP - 1)
    with pytest.raises(ValueError):
        features.observation(close, volume)
    close, volume = candles(features.SCALE_ROWS + features.WARMUP)
    assert features.observation(close, volume).shape == (30, 7)
