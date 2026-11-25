import pytest

from trader.market import MarketDataError, fetch_candles, parse_klines

NOW = 1_700_000_100_000  # ms


def kline(open_ms, close="100.5", volume="2.0"):
    return [open_ms, "100", "101", "99", close, volume, open_ms + 59_999, "0", 1, "0", "0", "0"]


def test_keeps_only_closed_candles():
    rows = [kline(NOW - 180_000), kline(NOW - 120_000), kline(NOW - 60_000), kline(NOW - 10_000)]
    c = parse_klines(rows, NOW)
    assert c.open_time_ms == [NOW - 180_000, NOW - 120_000, NOW - 60_000]
    assert c.close == [100.5] * 3 and c.volume == [2.0] * 3
    assert c.last_close_time_ms == NOW


@pytest.mark.parametrize("rows", [[kline(NOW - 120_000, close="0")], [kline(NOW - 120_000, close="nan")],
                                  [kline(NOW - 120_000, volume="-1")], [["x"]], {"code": -1121},
                                  [kline(NOW - 60_000), kline(NOW - 120_000)], [kline(NOW - 10_000)]])
def test_refuses_bad_data(rows):
    with pytest.raises(MarketDataError):
        parse_klines(rows, NOW)


def test_retries_then_falls_back_to_the_second_host():
    calls, sleeps = [], []

    def get_json(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise TimeoutError("timed out")
        return [kline(NOW - 120_000), kline(NOW - 60_000)]

    c = fetch_candles("BTCUSDT", 2, base_urls=("https://a", "https://b"), clock=lambda: NOW / 1000,
                      sleep=sleeps.append, get_json=get_json)
    assert len(c.close) == 2
    assert calls[0].startswith("https://a/api/v3/klines?") and calls[1].startswith("https://b/")
    assert "symbol=BTCUSDT" in calls[0] and "interval=1m" in calls[0] and "limit=2" in calls[0]
    assert sleeps == [2.0]


def test_gives_up_with_a_market_data_error():
    def get_json(url, timeout):
        raise OSError("network unreachable")

    with pytest.raises(MarketDataError):
        fetch_candles("BTCUSDT", 2, attempts=3, sleep=lambda s: None, get_json=get_json)
