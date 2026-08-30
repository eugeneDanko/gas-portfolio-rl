from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import IntEnum


class Regime(IntEnum):
    NORMAL = 0
    CRISIS = 1

    @property
    def label(self) -> str:
        return "normal" if self is Regime.NORMAL else "crisis"


@dataclass(frozen=True)
class MarketState:
    day: int
    gas_spot: float
    futures_price: float
    stock_price: float
    gas_factor: float
    regime: Regime
    maturity_day: int
    time_to_maturity: float
    true_gas_volatility: float
    calendar_date: date | None = None
    delivery_date: date | None = None
    forward_curve: tuple[float, ...] = ()
    forward_delivery_dates: tuple[date, ...] = ()
    stock_dividend_per_share: float = 0.0
    is_dividend_payment_date: bool = False


@dataclass(frozen=True)
class MarketTransition:
    previous: MarketState
    current: MarketState
    futures_settlement_price: float
    gas_log_return: float
    futures_log_return: float
    stock_log_return: float
    rolled: bool
