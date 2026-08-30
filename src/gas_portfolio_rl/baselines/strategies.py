from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from gas_portfolio_rl.config.models import (
    RelativeMomentumConfig,
    RollingHedgeConfig,
    SignedRollingHedgeConfig,
)


class ConstantAllocationBaseline:
    """A deterministic control strategy used for paired evaluation."""

    def __init__(
        self,
        stock_weight: float,
        futures_exposure: float,
        name: str,
        option_coverages: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rebalance_interval: int = 5,
    ) -> None:
        if len(option_coverages) != 3 or any(value < 0 for value in option_coverages):
            raise ValueError("option_coverages must contain three non-negative values")
        if sum(option_coverages) > 1.0 + 1e-12:
            raise ValueError("total option coverage cannot exceed 1")
        if rebalance_interval < 1:
            raise ValueError("rebalance_interval must be positive")
        self.name = name
        self.action = np.asarray([stock_weight, futures_exposure], dtype=np.float32)
        self.option_coverages = tuple(float(value) for value in option_coverages)
        self.rebalance_interval = rebalance_interval

    def reset(self) -> None:
        return None

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        del observation
        step = int(info["step"])
        option_enabled = bool(info.get("option_enabled", False))
        has_option_target = any(value > 0.0 for value in self.option_coverages)
        if step == 0:
            if option_enabled:
                return np.asarray(
                    [*self.action, *self.option_coverages], dtype=np.float32
                )
            return self.action.copy()
        if (
            option_enabled
            and has_option_target
            and step % self.rebalance_interval == 0
        ):
            return np.asarray(
                [
                    float(info["stock_weight"]),
                    float(info["futures_exposure"]),
                    *self.option_coverages,
                ],
                dtype=np.float32,
            )
        return _hold_current_positions(info)


class ProtectivePutBaseline:
    """Fixed stock allocation insured by one tenor or an equal-weight put ladder."""

    def __init__(
        self,
        *,
        stock_weight: float = 0.50,
        total_coverage: float = 1.0,
        tenor_allocation: tuple[float, float, float] = (0.0, 1.0, 0.0),
        rebalance_interval: int = 5,
        name: str = "protective_put",
    ) -> None:
        if not 0 <= stock_weight <= 1 or not 0 <= total_coverage <= 1:
            raise ValueError("stock_weight and total_coverage must be in [0, 1]")
        if len(tenor_allocation) != 3 or any(value < 0 for value in tenor_allocation):
            raise ValueError("tenor_allocation must contain three non-negative values")
        allocation_sum = sum(tenor_allocation)
        if allocation_sum <= 0 or rebalance_interval < 1:
            raise ValueError("tenor allocation and rebalance interval must be positive")
        self.name = name
        self.stock_weight = stock_weight
        self.total_coverage = total_coverage
        self.tenor_allocation = tuple(value / allocation_sum for value in tenor_allocation)
        self.rebalance_interval = rebalance_interval

    def reset(self) -> None:
        return None

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        del observation
        if not bool(info.get("option_enabled", False)):
            return np.asarray([self.stock_weight, 0.0], dtype=np.float32)
        if int(info["step"]) % self.rebalance_interval != 0:
            return _hold_current_positions(info)
        coverages = [self.total_coverage * value for value in self.tenor_allocation]
        return np.asarray([self.stock_weight, 0.0, *coverages], dtype=np.float32)


class RelativeMomentumBaseline:
    """Select stock or long futures by rolling dividend-adjusted momentum."""

    name = "relative_momentum"

    def __init__(
        self,
        config: RelativeMomentumConfig | None = None,
        *,
        trading_days: int = 252,
    ) -> None:
        self.config = config or RelativeMomentumConfig()
        self.trading_days = trading_days
        self.stock_prices: deque[float] = deque(maxlen=self.config.lookback + 1)
        self.futures_prices: deque[float] = deque(maxlen=self.config.lookback + 1)
        self.stock_dividends: deque[float] = deque(maxlen=self.config.lookback + 1)
        self.current_action = np.asarray(
            [self.config.neutral_stock_weight, 0.0], dtype=np.float32
        )
        self.last_scores = (0.0, 0.0)
        self._last_seen_step: int | None = None

    def reset(self) -> None:
        self.stock_prices.clear()
        self.futures_prices.clear()
        self.stock_dividends.clear()
        self.current_action = np.asarray(
            [self.config.neutral_stock_weight, 0.0], dtype=np.float32
        )
        self.last_scores = (0.0, 0.0)
        self._last_seen_step = None

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        del observation
        step = int(info["step"])
        self._append_prices_once(step, info)
        if step % self.config.rebalance_interval != 0:
            return _hold_current_positions(info)
        if len(self.stock_prices) < self.config.lookback + 1:
            return _adapt_action(self.current_action, info)

        stock_returns = _stock_total_returns(self.stock_prices, self.stock_dividends)
        futures_returns = np.diff(np.log(np.asarray(self.futures_prices)))
        stock_score = _risk_adjusted_momentum(stock_returns, self.trading_days)
        futures_score = _risk_adjusted_momentum(futures_returns, self.trading_days)
        self.last_scores = (stock_score, futures_score)

        if stock_score > futures_score + self.config.score_threshold:
            self.current_action = np.asarray([self.config.allocation, 0.0], dtype=np.float32)
        elif futures_score > stock_score + self.config.score_threshold:
            self.current_action = np.asarray([0.0, self.config.allocation], dtype=np.float32)
        else:
            self.current_action = np.asarray(
                [self.config.neutral_stock_weight, 0.0], dtype=np.float32
            )
        return _adapt_action(self.current_action, info)

    def _append_prices_once(self, step: int, info: Mapping[str, Any]) -> None:
        if step == self._last_seen_step:
            return
        self.stock_prices.append(float(info["stock_price"]))
        self.futures_prices.append(float(info["futures_price"]))
        self.stock_dividends.append(float(info.get("stock_dividend_per_share", 0.0)))
        self._last_seen_step = step


class RollingHedgeBaseline:
    """Hold GasCo and minimize variance with a rolling short futures hedge."""

    name = "rolling_hedge"

    def __init__(self, config: RollingHedgeConfig | None = None) -> None:
        self.config = config or RollingHedgeConfig()
        self.stock_prices: deque[float] = deque(maxlen=self.config.lookback + 1)
        self.futures_prices: deque[float] = deque(maxlen=self.config.lookback + 1)
        self.stock_dividends: deque[float] = deque(maxlen=self.config.lookback + 1)
        initial_exposure = -min(
            self.config.stock_weight * self.config.initial_hedge_ratio,
            self.config.maximum_futures_exposure,
        )
        self.current_action = np.asarray(
            [self.config.stock_weight, initial_exposure], dtype=np.float32
        )
        self.last_hedge_beta = self.config.initial_hedge_ratio
        self._last_seen_step: int | None = None

    def reset(self) -> None:
        self.stock_prices.clear()
        self.futures_prices.clear()
        self.stock_dividends.clear()
        self.last_hedge_beta = self.config.initial_hedge_ratio
        self.current_action = np.asarray(
            [
                self.config.stock_weight,
                -min(
                    self.config.stock_weight * self.config.initial_hedge_ratio,
                    self.config.maximum_futures_exposure,
                ),
            ],
            dtype=np.float32,
        )
        self._last_seen_step = None

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        del observation
        step = int(info["step"])
        self._append_prices_once(step, info)
        if step % self.config.rebalance_interval != 0:
            return _hold_current_positions(info)
        if len(self.stock_prices) < self.config.lookback + 1:
            return _adapt_action(self.current_action, info)

        stock_returns = _stock_total_returns(self.stock_prices, self.stock_dividends)
        futures_returns = np.diff(np.log(np.asarray(self.futures_prices)))
        futures_variance = float(np.var(futures_returns, ddof=1))
        beta = 0.0
        if futures_variance > 1e-12:
            covariance = float(np.cov(stock_returns, futures_returns, ddof=1)[0, 1])
            beta = max(0.0, covariance / futures_variance)
        self.last_hedge_beta = beta
        exposure = -min(
            beta * self.config.stock_weight,
            self.config.maximum_futures_exposure,
        )
        self.current_action = np.asarray(
            [self.config.stock_weight, exposure], dtype=np.float32
        )
        return _adapt_action(self.current_action, info)

    def _append_prices_once(self, step: int, info: Mapping[str, Any]) -> None:
        if step == self._last_seen_step:
            return
        self.stock_prices.append(float(info["stock_price"]))
        self.futures_prices.append(float(info["futures_price"]))
        self.stock_dividends.append(float(info.get("stock_dividend_per_share", 0.0)))
        self._last_seen_step = step


class SignedRollingHedgeBaseline(RollingHedgeBaseline):
    """Minimum-variance hedge that can become short or long futures."""

    name = "signed_rolling_hedge"

    def __init__(self, config: SignedRollingHedgeConfig | None = None) -> None:
        super().__init__(config or SignedRollingHedgeConfig())
        self.config: SignedRollingHedgeConfig
        self.last_raw_beta = self.config.initial_hedge_ratio
        self.last_correlation = 0.0
        self._has_beta_estimate = False

    def reset(self) -> None:
        super().reset()
        self.last_raw_beta = self.config.initial_hedge_ratio
        self.last_correlation = 0.0
        self._has_beta_estimate = False

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        del observation
        step = int(info["step"])
        self._append_prices_once(step, info)
        if step % self.config.rebalance_interval != 0:
            return _hold_current_positions(info)
        if len(self.stock_prices) < self.config.lookback + 1:
            return _adapt_action(self.current_action, info)

        stock_returns = _stock_total_returns(self.stock_prices, self.stock_dividends)
        futures_returns = np.diff(np.log(np.asarray(self.futures_prices)))
        futures_variance = float(np.var(futures_returns, ddof=1))
        raw_beta = 0.0
        correlation = 0.0
        if futures_variance > 1e-12 and np.std(stock_returns, ddof=1) > 1e-12:
            covariance = float(np.cov(stock_returns, futures_returns, ddof=1)[0, 1])
            raw_beta = covariance / futures_variance
            correlation = float(np.corrcoef(stock_returns, futures_returns)[0, 1])
        self.last_raw_beta = raw_beta
        self.last_correlation = correlation

        shrunk_beta = (
            raw_beta * self.config.beta_shrinkage
            if abs(correlation) >= self.config.minimum_abs_correlation
            else 0.0
        )
        if self._has_beta_estimate:
            smoothed_beta = (
                (1.0 - self.config.beta_smoothing) * self.last_hedge_beta
                + self.config.beta_smoothing * shrunk_beta
            )
        else:
            smoothed_beta = shrunk_beta
            self._has_beta_estimate = True
        self.last_hedge_beta = smoothed_beta

        target_exposure = float(np.clip(
            -smoothed_beta * self.config.stock_weight,
            -self.config.maximum_futures_exposure,
            self.config.maximum_futures_exposure,
        ))
        if (
            abs(target_exposure - float(self.current_action[1]))
            < self.config.exposure_change_threshold
        ):
            return _hold_current_positions(info)
        self.current_action = np.asarray(
            [self.config.stock_weight, target_exposure], dtype=np.float32
        )
        return _adapt_action(self.current_action, info)


def _risk_adjusted_momentum(returns: np.ndarray, trading_days: int) -> float:
    volatility = float(np.std(returns, ddof=1))
    return float(np.mean(returns) / max(volatility, 1e-8) * np.sqrt(trading_days))


def _stock_total_returns(
    prices: deque[float], dividends: deque[float]
) -> np.ndarray:
    price_values = np.asarray(prices)
    dividend_values = np.asarray(dividends)
    return np.log((price_values[1:] + dividend_values[1:]) / price_values[:-1])


def _hold_current_positions(info: Mapping[str, Any]) -> np.ndarray:
    values = [float(info.get("stock_weight", 0.0)), float(info.get("futures_exposure", 0.0))]
    if bool(info.get("option_enabled", False)):
        values.extend(float(value) for value in info.get("option_coverages", (0, 0, 0)))
    return np.asarray(values, dtype=np.float32)


def _adapt_action(action: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
    values = list(np.asarray(action, dtype=np.float32).reshape(-1))
    if bool(info.get("option_enabled", False)) and len(values) == 2:
        values.extend([0.0, 0.0, 0.0])
    return np.asarray(values, dtype=np.float32)
