import pytest

from trader.portfolio import FEE, Portfolio, apply, buy_and_hold_units


def test_buy_spends_all_cash_and_pays_the_fee_in_btc():
    p = Portfolio(1000.0, 0.0)
    t = apply(p, 0, 50_000.0)
    assert t.side == "buy"
    assert p.cash == 0.0
    assert p.crypto == pytest.approx(1000 / 50_000 * (1 - FEE))
    assert t.quantity == pytest.approx(p.crypto)
    assert t.fee_usdt == pytest.approx(1.0)  # 0.1% of 1000 USDT
    assert p.value(50_000.0) == pytest.approx(999.0)


def test_sell_sells_everything_and_pays_the_fee_in_usdt():
    p = Portfolio(0.0, 0.02)
    t = apply(p, 1, 60_000.0)
    assert t.side == "sell" and t.quantity == 0.02
    assert p.crypto == 0.0
    assert p.cash == pytest.approx(1200 * (1 - FEE))
    assert t.fee_usdt == pytest.approx(1.2)


def test_round_trip_at_the_same_price_costs_two_fees():
    p = Portfolio(1000.0, 0.0)
    apply(p, 0, 42_000.0)
    apply(p, 1, 42_000.0)
    assert p.cash == pytest.approx(1000 * (1 - FEE) ** 2)


@pytest.mark.parametrize("portfolio,action", [(Portfolio(0.0, 0.1), 0), (Portfolio(500.0, 0.0), 1),
                                              (Portfolio(500.0, 0.0), 2), (Portfolio(0.0, 0.1), 2)])
def test_impossible_or_hold_changes_nothing(portfolio, action):
    before = (portfolio.cash, portfolio.crypto)
    assert apply(portfolio, action, 30_000.0) is None
    assert (portfolio.cash, portfolio.crypto) == before


def test_buy_and_hold_pays_one_fee():
    units = buy_and_hold_units(1000.0, 40_000.0)
    assert units * 40_000.0 == pytest.approx(999.0)


def test_bad_price_is_refused():
    with pytest.raises(ValueError):
        apply(Portfolio(1000.0, 0.0), 0, 0.0)


def test_fixed_size_buy_spends_that_amount_and_pays_the_fee_in_btc():
    p = Portfolio(1000.0, 0.0)
    t = apply(p, 0, 50_000.0, amount=25.0)
    assert p.cash == 975.0
    assert p.crypto == pytest.approx(25 / 50_000 * (1 - FEE)) and t.quantity == p.crypto
    assert t.fee_usdt == pytest.approx(0.025)
    apply(p, 0, 40_000.0, amount=25.0)  # a second one adds to the position
    assert p.cash == 950.0 and p.crypto == pytest.approx((25 / 50_000 + 25 / 40_000) * (1 - FEE))


def test_fixed_size_buy_larger_than_the_cash_spends_what_is_left():
    p = Portfolio(10.0, 0.5)
    t = apply(p, 0, 50_000.0, amount=25.0)
    assert p.cash == 0.0 and t.fee_usdt == pytest.approx(0.01)
    assert apply(p, 0, 50_000.0, amount=25.0) is None  # nothing left
    assert apply(Portfolio(1e-9, 0.0), 0, 50_000.0, amount=25.0) is None  # dust is not traded
