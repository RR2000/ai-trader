"""The paper portfolio: all-in buys and sells as in the 2023 simulators, with exchange fees."""

from __future__ import annotations

from dataclasses import dataclass

START_CASH = 1000.0  # USDT
FEE = 0.001  # 0.1% per trade: Binance spot base rate without BNB discount


@dataclass
class Portfolio:
    cash: float  # USDT
    crypto: float  # BTC
    fee: float = FEE

    def value(self, price: float) -> float:
        """Marked to the last price (what it is worth before paying a sell fee)."""
        return self.cash + self.crypto * price


@dataclass(frozen=True)
class Trade:
    side: str  # "buy" or "sell"
    price: float
    quantity: float  # BTC bought (after the fee) or sold
    fee_usdt: float  # fee paid, in USDT


def apply(portfolio: Portfolio, action: int, price: float) -> Trade | None:
    """0 = buy with all the cash, 1 = sell all the BTC, 2 = hold. A buy with no cash (or a sell with
    no BTC) does nothing, as in TradingSimulator.py. The fee is taken from what is received."""
    if price <= 0:
        raise ValueError("price must be positive")
    if action == 0 and portfolio.cash > 0:
        gross = portfolio.cash / price
        fee_btc = gross * portfolio.fee
        trade = Trade("buy", price, gross - fee_btc, fee_btc * price)
        portfolio.crypto += gross - fee_btc
        portfolio.cash = 0.0
        return trade
    if action == 1 and portfolio.crypto > 0:
        gross = portfolio.crypto * price
        fee = gross * portfolio.fee
        trade = Trade("sell", price, portfolio.crypto, fee)
        portfolio.cash += gross - fee
        portfolio.crypto = 0.0
        return trade
    return None


def buy_and_hold_units(cash: float, price: float, fee: float = FEE) -> float:
    """BTC that the same cash buys at the start, paying one buy fee: the honest benchmark."""
    return cash / price * (1.0 - fee)
