from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HistoryFeatures:
    gas_return: float
    stock_return: float
    futures_return: float
    stock_volatility: float
    futures_volatility: float
    stock_futures_correlation: float


class HistoryBuffer:
    def __init__(self, window: int, trading_days: int) -> None:
        self.window = window
        self.trading_days = trading_days
        self.gas_returns: deque[float] = deque(maxlen=window)
        self.stock_returns: deque[float] = deque(maxlen=window)
        self.futures_returns: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self.gas_returns.clear()
        self.stock_returns.clear()
        self.futures_returns.clear()

    def append(self, gas_return: float, stock_return: float, futures_return: float) -> None:
        self.gas_returns.append(float(gas_return))
        self.stock_returns.append(float(stock_return))
        self.futures_returns.append(float(futures_return))

    def features(self) -> HistoryFeatures:
        if not self.gas_returns:
            return HistoryFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        stock = np.asarray(self.stock_returns)
        future = np.asarray(self.futures_returns)
        stock_vol = _annualized_volatility(stock, self.trading_days)
        future_vol = _annualized_volatility(future, self.trading_days)
        correlation = 0.0
        if len(stock) >= 2 and np.std(stock) > 1e-12 and np.std(future) > 1e-12:
            correlation = float(np.corrcoef(stock, future)[0, 1])
        return HistoryFeatures(
            self.gas_returns[-1],
            self.stock_returns[-1],
            self.futures_returns[-1],
            stock_vol,
            future_vol,
            correlation,
        )

    def portfolio_volatility(self, stock_weight: float, futures_exposure: float) -> float:
        """Annualized volatility using only returns observable up to the current step."""
        if len(self.stock_returns) < 2:
            return 0.0
        stock = np.asarray(self.stock_returns)
        future = np.asarray(self.futures_returns)
        portfolio_returns = stock_weight * stock + futures_exposure * future
        return _annualized_volatility(portfolio_returns, self.trading_days)


def _annualized_volatility(values: np.ndarray, trading_days: int) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) * np.sqrt(trading_days))
