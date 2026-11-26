"""Train the small network of simulation mlp-2026 and evaluate it honestly, walk-forward.

    pip install --no-deps -r requirements-dev.txt
    python training/train_mlp.py            # about half an hour on a laptop CPU

1. Downloads BTC/USDT one-hour candles from Binance's public archive (data.binance.vision, monthly files,
   checked against their SHA-256) from January 2019 to the last complete month, into training/.cache/.
2. Features (trader/mlfeatures.py, the same code the live service runs): returns over 1 to 168 hours,
   volatility, RSI, distance from two moving averages, volume. Target: is the close 24 hours later higher?
3. Walk-forward: for every quarter from 2021 on, a network is trained only on the hours before that quarter
   (the last 24 are dropped, their target would look into it) and trades that quarter, never seeing it before.
   The settings below were fixed before the first run and were not tuned on the results.
4. The same quarters are traded by buy and hold and the classic strategies of trader/strategies.py, with the
   same accounting (0.1% fee, no slippage), so the comparison is like for like.
5. Writes training/results.json and the table for the README, and the final model, trained on everything up to
   the last complete month, to src/model/mlp/.

Fixed seeds: the same data and library versions (requirements-dev.txt) should give the same numbers.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trader import mlfeatures  # noqa: E402
from trader.market import Candles  # noqa: E402
from trader.portfolio import Portfolio, apply, buy_and_hold_units  # noqa: E402
from trader.strategies import Position, build  # noqa: E402

ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{}.zip"
CACHE = Path(__file__).parent / ".cache"
HOUR = 3_600_000
FIRST_MONTH = (2019, 1)
FIRST_TEST = (2021, 1)
# fixed in advance, not tuned on the test quarters
HORIZON = 24  # hours ahead of the target
HIDDEN = (16,)
ALPHA = 1e-3  # L2 penalty
EPOCHS = 60
SEED = 0
ENTER, LEAVE = 0.55, 0.45
FEE = 0.001
CLASSICS = {
    "buy-and-hold": ("buy-and-hold", {}),
    "sma-cross 20/50": ("sma-cross", {"fast": 20, "slow": 50}),
    "rsi 14 30/70": ("rsi", {"period": 14, "oversold": 30, "overbought": 70}),
    "bollinger 20/2": ("bollinger", {"period": 20, "width": 2.0}),
    "dca 25/day": ("dca", {"amount": 25, "every": "1d"}),
}


def months(last: tuple[int, int]):
    y, m = FIRST_MONTH
    while (y, m) <= last:
        yield f"{y}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-trader-training"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def month_rows(month: str) -> list[tuple[int, float, float]]:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"BTCUSDT-1h-{month}.zip"
    if not path.exists():
        data = fetch(ARCHIVE.format(month))
        expected = fetch(ARCHIVE.format(month) + ".CHECKSUM").split()[0].decode()
        if hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit(f"checksum mismatch for {month}")
        path.write_bytes(data)
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode()
    rows = []
    for r in csv.reader(io.StringIO(text)):
        if not r or not r[0].isdigit():
            continue  # a header line in some files
        t = int(r[0])
        if t > 10**14:
            t //= 1000  # the archive moved to microseconds in 2025
        rows.append((t, float(r[4]), float(r[5])))
    return rows


def load(last_month: tuple[int, int]):
    rows = sorted({r[0]: r for m in months(last_month) for r in month_rows(m)}.values())
    t = np.array([r[0] for r in rows], dtype=np.int64)
    return t, np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def ms(y: int, m: int) -> int:
    return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)


def train(x: np.ndarray, y: np.ndarray) -> mlfeatures.MLP:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier

    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-12] = 1.0
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, alpha=ALPHA, max_iter=EPOCHS, batch_size=256,
                        learning_rate_init=1e-3, random_state=SEED, shuffle=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit((x - mean) / std, y)
    meta = {"features": list(mlfeatures.NAMES), "horizon_hours": HORIZON, "layers": len(clf.coefs_),
            "hidden": list(HIDDEN), "alpha": ALPHA, "epochs": EPOCHS, "seed": SEED, "rows": int(len(y))}
    return mlfeatures.MLP(mean, std, tuple(clf.coefs_), tuple(c for c in clf.intercepts_), meta), clf


def run(decide, t, close, start: int, end: int, cash=1000.0):
    """Trade hours start..end-1 (indexes): decide(i, position) -> (action, amount). Value at every hour."""
    p, last_buy, trades, fees = Portfolio(cash, 0.0, FEE), None, 0, 0.0
    values = np.empty(end - start)
    for k, i in enumerate(range(start, end)):
        a, amount = decide(i, Position(p.cash, p.crypto, last_buy))
        tr = apply(p, a, close[i], amount)
        if tr:
            trades += 1
            fees += tr.fee_usdt
            if tr.side == "buy":
                last_buy = datetime.fromtimestamp((t[i] + HOUR) / 1000, tz=timezone.utc)
        values[k] = p.value(close[i])
    return values, trades, fees


def classic(kind, params, t, close, volume):
    s = build(kind, params)
    n = s.lookback

    def decide(i, pos):
        c = Candles(list(t[i - n + 1:i + 1]), list(close[i - n + 1:i + 1]), list(volume[i - n + 1:i + 1]), HOUR)
        d = s.decide(c, pos)
        return d.action, d.amount
    return decide


def summary(values, t_idx, t, trades, fees, start_cash=1000.0):
    years = {}
    for k, i in enumerate(t_idx):
        y = datetime.fromtimestamp(t[i] / 1000, tz=timezone.utc).year
        years.setdefault(y, [k, k])[1] = k
    by_year = {}
    prev = start_cash
    for y, (a, b) in years.items():
        by_year[str(y)] = round((values[b] / prev - 1) * 100, 1)
        prev = values[b]
    peak = np.maximum.accumulate(values)
    return {"final": round(float(values[-1]), 2), "return_pct": round((values[-1] / start_cash - 1) * 100, 1),
            "max_drawdown_pct": round(float(((values - peak) / peak).min() * 100), 1), "trades": trades,
            "fees": round(fees, 2), "by_year": by_year}


def main():
    now = datetime.now(timezone.utc)
    last = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
    t, close, volume = load(last)
    print(f"{len(t)} hourly candles, {datetime.fromtimestamp(t[0] / 1000, tz=timezone.utc):%Y-%m-%d} to "
          f"{datetime.fromtimestamp(t[-1] / 1000, tz=timezone.utc):%Y-%m-%d}", flush=True)
    feats = mlfeatures.table(close, volume)
    target = np.full(len(close), np.nan)
    target[:-HORIZON] = (close[HORIZON:] > close[:-HORIZON]).astype(float)
    ok = ~np.isnan(feats).any(axis=1)

    # walk-forward: one network per quarter, trained on what came before it only
    test_start = int(np.searchsorted(t, ms(*FIRST_TEST)))
    proba = np.full(len(t), np.nan)
    quarters, accuracy = [], []
    y, m = FIRST_TEST
    while ms(y, m) <= t[-1]:
        ny, nm = (y + 1, m - 9) if m > 9 else (y, m + 3)
        a, b = int(np.searchsorted(t, ms(y, m))), int(np.searchsorted(t, ms(ny, nm)))
        train_rows = ok.copy()
        train_rows[a - HORIZON:] = False  # purge: these targets reach into the test quarter
        train_rows &= ~np.isnan(target)
        model, _ = train(feats[train_rows], target[train_rows])
        proba[a:b] = model.proba(feats[a:b])
        hit = (proba[a:b] > 0.5) == (target[a:b] == 1)
        valid = ~np.isnan(target[a:b])
        accuracy.append(float(hit[valid].mean()))
        quarters.append(f"{y}-{m:02d}")
        print(f"quarter {y}-{m:02d}: trained on {train_rows.sum()} hours, test {b - a}, "
              f"accuracy {accuracy[-1]:.3f}", flush=True)
        y, m = ny, nm

    end = len(t)
    idx = range(test_start, end)
    results = {}
    values, trades, fees = run(lambda i, pos: (mlfeatures.action(proba[i], ENTER, LEAVE), None), t, close, test_start, end)
    results["mlp (walk-forward)"] = summary(values, idx, t, trades, fees)
    for name, (kind, params) in CLASSICS.items():
        v, tr, fe = run(classic(kind, params, t, close, volume), t, close, test_start, end)
        results[name] = summary(v, idx, t, tr, fe)
        print(name, results[name]["return_pct"], flush=True)
    bh_check = buy_and_hold_units(1000.0, close[test_start], FEE) * close[-1]
    assert abs(results["buy-and-hold"]["final"] - bh_check) < 0.01

    out = {
        "data": {"symbol": "BTCUSDT", "interval": "1h", "first": f"{FIRST_MONTH[0]}-{FIRST_MONTH[1]:02d}",
                 "last": f"{last[0]}-{last[1]:02d}", "candles": int(len(t))},
        "test": {"from": f"{FIRST_TEST[0]}-{FIRST_TEST[1]:02d}", "to": f"{last[0]}-{last[1]:02d}",
                 "quarters": len(quarters), "accuracy_mean": round(float(np.mean(accuracy)), 4),
                 "accuracy_by_quarter": dict(zip(quarters, [round(a, 4) for a in accuracy])),
                 "share_up_24h": round(float(np.nanmean(target[test_start:])), 4)},
        "settings": {"horizon_hours": HORIZON, "hidden": list(HIDDEN), "alpha": ALPHA, "epochs": EPOCHS,
                     "seed": SEED, "enter": ENTER, "leave": LEAVE, "fee": FEE},
        "results": results,
    }
    (Path(__file__).parent / "results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\n| strategy | return | max drawdown | trades | fees | " + " | ".join(results["buy-and-hold"]["by_year"]) + " |")
    print("|---|---:|---:|---:|---:|" + "---:|" * len(results["buy-and-hold"]["by_year"]))
    for name, r in results.items():
        print(f"| {name} | {r['return_pct']:+.1f}% | {r['max_drawdown_pct']:.1f}% | {r['trades']} | {r['fees']:,.0f} | "
              + " | ".join(f"{v:+.1f}%" for v in r["by_year"].values()) + " |")

    # the model that runs live: everything up to the last complete month
    rows = ok & ~np.isnan(target)
    model, clf = train(feats[rows], target[rows])
    model.meta.update({"trained_until": f"{last[0]}-{last[1]:02d}", "enter": ENTER, "leave": LEAVE})
    dest = ROOT / "src" / "model" / "mlp"
    dest.mkdir(parents=True, exist_ok=True)
    name = f"mlp_btcusdt_1h_{last[0]}-{last[1]:02d}.npz"
    model.save(dest / name)
    check = feats[rows][-500:]
    np.testing.assert_allclose(model.proba(check), clf.predict_proba((check - model.mean) / model.std)[:, 1], atol=1e-9)
    print(f"\nlive model: src/model/mlp/{name}")


if __name__ == "__main__":
    main()
