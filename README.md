# ai-trader

A deep Q-network that trades Bitcoin, written in April 2023 as a reinforcement-learning experiment,
now running live next to a few classic strategies, each on its own **simulated** portfolio:
**https://trader.raoulrondinella.dev**

The home page compares them all: value, return, the difference from "buy and hold" over the same period,
number of trades and when each was last updated. Each simulation has its own page (`/s/<id>`) with the value
over time next to buy and hold from the same moment, every trade and how the strategy decides. It is an
honest comparison: if a strategy loses money, the page shows it. No real money is involved and nothing here
is financial advice.

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

## The simulations (`simulations.toml`)

The simulations are defined only in [`simulations.toml`](simulations.toml), in this repository: nothing can be
created from the site. The file is read once at start-up and checked; any mistake (an unknown strategy, a
misspelled key, a parameter out of range, an id used twice) stops the service with a message naming the
simulation and the field, and the tests check the shipped file, so CI catches it before a deploy.

### Adding one

Add one block and deploy:

```toml
[[simulation]]
id = "rsi-eth-1h"                 # lowercase letters, digits, dashes: the page is /s/rsi-eth-1h
name = "RSI 14 on ETH, hourly"
description = "Mean reversion on ETH/USDT."
strategy = "rsi"
symbol = "ETHUSDT"                # optional, default from [defaults]: BTCUSDT
interval = "1h"                   # optional: 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d
start_cash = 1000                 # optional, in the quote currency (USDT)
fee = 0.001                       # optional, 0.1% per trade
[simulation.params]
period = 14
oversold = 30
overbought = 70
```

It starts at its first candle with its own cash and is compared with buy and hold from that moment. Any Binance
spot pair works; the quote currency (USDT, USDC, FDUSD, EUR, BTC, ...) is the cash. Once a simulation has
started, its `strategy`, `params`, `symbol`, `interval`, `start_cash` and `fee` are fixed (they are stored
with it): if they change, that simulation stops and its page says so, the others go on. To try other settings,
add a new id. `name` and `description` can change any time. Removing a block hides the simulation; its history
stays in the database.

### Strategies and parameters

All of them use the same accounting: buy with all the cash (DCA: a fixed amount), sell everything, or hold; the
fee taken from what is received; no slippage; buy and hold from the same first candle, paying one fee, as the
benchmark. A strategy only sees closed candles and the stored position, so a restart changes nothing.

| strategy | what it does | parameters (default) |
|---|---|---|
| `dqn` | the 2023 deep Q-network, see below; it needs one-minute candles to make sense | `model` (`trading_model_episode_30.h5`): file in `MODEL_DIR`; the image contains only this one |
| `sma-cross` | trend following: holds the coin while the fast simple moving average is above the slow one, sells when it drops below | `fast` (20), `slow` (50): candles, fast < slow |
| `rsi` | mean reversion on the RSI (Wilder smoothing, as `ta`): buys below `oversold`, sells above `overbought` | `period` (14), `oversold` (30), `overbought` (70) |
| `bollinger` | mean reversion on Bollinger Bands: buys on a close below the lower band, sells above the upper band | `period` (20): candles of the average; `width` (2.0): standard deviations |
| `dca` | dollar-cost averaging: buys a fixed amount at fixed times until the cash runs out, never sells | `amount` (25): quote currency per buy; `every` (`1d`): like `30m`, `12h`, `7d` |
| `buy-and-hold` | buys with all the cash at the first candle and never sells: the reference | none |

The shipped file starts with the 2023 DQN (`dqn-2023`, one-minute candles) and five classic ones on BTC/USDT,
1,000 USDT each: buy and hold (15 min), SMA crossover 20/50 (1 h), RSI 14 30/70 (15 min), Bollinger 20/2
(15 min) and daily DCA of 25 USDT (1 h candles).

## The live service (`trader/`)

Once a minute, a few seconds after a candle closes, the service:

1. for every pair and interval with a new closed candle, reads the latest closed candles once from Binance's
   public market data (`data-api.binance.vision`, no API key; timeouts, retries and a fallback host) and hands
   them to every simulation that uses them, each getting exactly the history its strategy needs;
2. lets each simulation decide and applies the decision to its paper portfolio;
3. writes the decision, the trade and the balances to PostgreSQL in one transaction per simulation, so a restart
   or a new deploy continues the same portfolios and a candle is never used twice.

If Binance or the database does not answer, that candle is skipped and the pages say the data is not updated
(a simulation is "not updated" when its next candle is more than two minutes late). A simulation whose strategy
fails is skipped and retried on the next pass; the others are not affected.

### `dqn-2023`: the 2023 model

Every minute:

1. the latest 500 one-minute candles are requested, as before, and the 499 closed ones are used;
2. the observation is built exactly as in training: Close, Volume, SMA 10, SMA 30, RSI, MACD, MACD signal,
   min-max scaled to [0.01, 1.01], the last 30 rows;
3. `trading_model_episode_30.h5` (the model the 2023 live script used) runs and the action with the highest
   Q-value is taken, no exploration;
4. all in or all out as in `TradingSimulator.py`: started with 1,000 USDT, 0.1% fee per trade (Binance's base
   spot rate), no slippage.

This was the only simulation of the first release; its whole history (portfolio, decisions, trades) carries on
under `dqn-2023` (see *Database* below).

#### What had to change from 2023

- **BTCBUSD -> BTCUSDT.** Binance delisted BUSD.
- **No look-ahead in the scaling.** In training the min-max scaler was fit on the whole 107-row episode, so each
  window was scaled knowing the future; the 2023 live script fit it on the last week instead. Live there is no
  future: the scaler is fit on the last 107 rows (the training episode length), ending at the latest candle.
- **Closed candles only.** The 2023 live script appended the still-forming candle every minute.
- **No TensorFlow at run time.** The network is small, so `trader/model.py` reads the weights from the `.h5`
  file with h5py and runs the forward pass in NumPy. The tests check it against Keras (TensorFlow 2.21 reference
  outputs, plus Keras 3 with the JAX backend in CI) and the observation against `ta`, pandas and scikit-learn,
  the libraries the model was trained with.

### Database

Tables `sim_portfolio` (one row per simulation, with the definition it started with), `sim_equity` (one row per
decision) and `sim_trades`. The first release had a single simulation in the tables `portfolio`, `equity` and
`trades`: at the first start, the new version copies them, row by row, into the new tables as `dqn-2023`, in one
transaction (all or nothing) recorded in `schema_migrations`, so it never runs twice. The old tables are left
untouched: they are a backup, and the previous image still runs on them if a deploy is rolled back (the minutes
it trades there are not carried over again). They can be dropped once the new version has run for a while.

### Run it

Python 3.14 and a PostgreSQL database:

```bash
pip install --no-deps -r requirements.txt
DB_HOST=localhost DB_NAME=trader DB_USER=trader DB_PASSWORD=... python -m trader   # http://localhost:8080
```

Settings (environment): `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` (or `DATABASE_URL`),
`SIMULATIONS` (default `simulations.toml` in the repository), `MODEL_DIR` (default `src/model/completed`),
`PORT` (default 8080), `LOG_LEVEL`.

Pages: `/` (comparison), `/s/<id>` (one simulation). JSON: `/api/simulations`,
`/api/simulations/<id>?range=1d|7d|30d|all`, and `/api/state?range=...` (the first release's address) for `dqn-2023`.

Container image: `ghcr.io/rr2000/ai-trader:main` (and `:sha-<commit>`), built by CI on every push to `main`.

### Tests

```bash
pip install --no-deps -r requirements-dev.txt
KERAS_BACKEND=jax python -m pytest
```

The persistence tests need PostgreSQL: `TEST_DATABASE_URL`, or `initdb` and `pg_ctl` on the PATH for a
throwaway local cluster. `tests/make_keras_reference.py` regenerates the TensorFlow reference outputs.
