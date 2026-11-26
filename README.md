# ai-trader

A deep Q-network that trades Bitcoin, written in April 2023 as a reinforcement-learning experiment,
now running live on a **simulated** portfolio: **https://trader.raoulrondinella.dev**

The page shows the portfolio value over time next to "buy and hold" from the same moment, every trade
and when the data was last updated. It is an honest comparison: if the model loses money, the page
shows it. No real money is involved and nothing here is financial advice.

## The 2023 experiment

| file | what it does |
|---|---|
| `CSVDownloader.py` | downloads one-minute BTC candles from Binance into a CSV |
| `TradingEnvironment.py` | a Gym environment: observation = the last 30 rows, actions buy / sell / hold, reward = change in portfolio value |
| `DQN.py` | the agent: LSTM(64) -> LSTM(32) -> Dense(16, relu) -> Dense(3), epsilon-greedy, experience replay |
| `ModelTrainer.py` | adds SMA 10, SMA 30, RSI 14, MACD and MACD signal (`ta`), min-max scales to [0.01, 1.01], trains |
| `TradingSimulator.py` | replays random two-week slices of history against buy and hold |
| `BinanceTradingSimulator.py` | the first attempt at running it on live prices |
| `src/model/completed/*.h5` | the trained Keras models |

Be warned: the training used a tiny slice of data (the last 107 minutes of the downloaded year, 200 episodes
over the same rows), so the model has seen very little of the market. That is part of the experiment.

These files are kept as they were, apart from the Binance API key lines: market data is public and needs no key.

## The live demo (`trader/`)

Once a minute, a few seconds after a candle closes, the service:

1. reads the latest 500 closed one-minute BTC/USDT candles from Binance's public market data
   (`data-api.binance.vision`, no API key; timeouts, retries and a fallback host);
2. builds the observation exactly as in training: Close, Volume, SMA 10, SMA 30, RSI, MACD, MACD signal,
   min-max scaled to [0.01, 1.01], the last 30 rows;
3. runs `trading_model_episode_30.h5` (the model the 2023 live script used) and takes the action with the
   highest Q-value, no exploration;
4. applies it to the paper portfolio, all in or all out as in `TradingSimulator.py`: started with 1,000 USDT,
   0.1% fee per trade (Binance's base spot rate), no slippage. Buy and hold spent the same 1,000 USDT at the
   first price, paying one fee;
5. writes the decision, the trade and the balances to PostgreSQL in one transaction, so a restart or a new
   deploy continues the same portfolio and a candle is never used twice.

If Binance or the database does not answer, the minute is skipped and the page says the data is not updated.

### What had to change from 2023

- **BTCBUSD -> BTCUSDT.** Binance delisted BUSD.
- **No look-ahead in the scaling.** In training the min-max scaler was fit on the whole 107-row episode, so each
  window was scaled knowing the future; the 2023 live script fit it on the last week instead. Live there is no
  future: the scaler is fit on the last 107 rows (the training episode length), ending at the latest candle.
- **Closed candles only.** The 2023 live script appended the still-forming candle every minute.
- **No TensorFlow at run time.** The network is small, so `trader/model.py` reads the weights from the `.h5`
  file with h5py and runs the forward pass in NumPy. The tests check it against Keras (TensorFlow 2.21 reference
  outputs, plus Keras 3 with the JAX backend in CI) and the observation against `ta`, pandas and scikit-learn,
  the libraries the model was trained with.

### Run it

Python 3.14 and a PostgreSQL database:

```bash
pip install --no-deps -r requirements.txt
DB_HOST=localhost DB_NAME=trader DB_USER=trader DB_PASSWORD=... python -m trader   # http://localhost:8080
```

Settings (environment): `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` (or `DATABASE_URL`),
`SYMBOL` (default `BTCUSDT`), `MODEL_PATH`, `PORT` (default 8080), `LOG_LEVEL`.

Container image: `ghcr.io/rr2000/ai-trader:main` (and `:sha-<commit>`), built by CI on every push to `main`.

### Tests

```bash
pip install --no-deps -r requirements-dev.txt
KERAS_BACKEND=jax python -m pytest
```

The persistence tests need PostgreSQL: `TEST_DATABASE_URL`, or `initdb` and `pg_ctl` on the PATH for a
throwaway local cluster. `tests/make_keras_reference.py` regenerates the TensorFlow reference outputs.
