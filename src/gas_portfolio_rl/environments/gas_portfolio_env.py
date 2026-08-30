from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.environments.actions import ProjectedAction, project_action
from gas_portfolio_rl.environments.history import HistoryBuffer
from gas_portfolio_rl.environments.observations import (
    OBSERVATION_HIGH,
    OBSERVATION_LOW,
    OBSERVATION_NAMES,
    OPTION_OBSERVATION_HIGH,
    OPTION_OBSERVATION_LOW,
    OPTION_OBSERVATION_NAMES,
    build_observation,
)
from gas_portfolio_rl.environments.rewards import calculate_reward
from gas_portfolio_rl.portfolio.accounting import AccountingResult, Portfolio, RebalanceResult
from gas_portfolio_rl.simulation.market_simulator import MarketSimulator
from gas_portfolio_rl.simulation.calendar import next_contract_date
from gas_portfolio_rl.simulation.models import MarketState


class GasPortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Causal Gymnasium environment: observe -> act -> simulate -> account."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, config: AppConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config or AppConfig()
        if render_mode not in (None, "ansi"):
            raise ValueError("Only render_mode='ansi' is supported")
        self.render_mode = render_mode
        self.market = MarketSimulator(self.config.market)
        self.portfolio = Portfolio(
            self.config.portfolio,
            self.config.market.trading_days,
            self.config.option,
            self.config.market.annual_stock_dividend_yield,
        )
        self.history = HistoryBuffer(
            self.config.environment.rolling_window,
            self.config.market.trading_days,
        )
        action_low = [0.0, -1.0]
        action_high = [1.0, 1.0]
        if self.config.option.enabled:
            action_low.extend([0.0, 0.0, 0.0])
            action_high.extend([1.0, 1.0, 1.0])
        self.action_space = spaces.Box(
            low=np.asarray(action_low, dtype=np.float32),
            high=np.asarray(action_high, dtype=np.float32),
            dtype=np.float32,
        )
        observation_low = OBSERVATION_LOW
        observation_high = OBSERVATION_HIGH
        if self.config.option.enabled:
            observation_low = np.concatenate((observation_low, OPTION_OBSERVATION_LOW))
            observation_high = np.concatenate((observation_high, OPTION_OBSERVATION_HIGH))
        if self.config.environment.observable_regime:
            observation_low = np.append(observation_low, np.float32(0.0))
            observation_high = np.append(observation_high, np.float32(1.0))
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )
        self.elapsed_steps = 0
        self.cumulative_reward = 0.0
        self.current_market: MarketState | None = None
        self.trajectory: list[dict[str, Any]] = []

    @property
    def observation_names(self) -> tuple[str, ...]:
        names = OBSERVATION_NAMES
        if self.config.option.enabled:
            names += OPTION_OBSERVATION_NAMES
        if self.config.environment.observable_regime:
            names += ("true_regime",)
        return names

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.action_space.seed(seed)
        self.current_market = self.market.reset(seed)
        self.portfolio.reset(self.current_market)
        self.history.reset()
        self.elapsed_steps = 0
        self.cumulative_reward = 0.0
        info = self._info(
            action=None,
            rebalance=None,
            accounting=None,
            reward_components=None,
        )
        self.trajectory = [dict(info)]
        return self._observation(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_market is None:
            raise RuntimeError("Call reset() before step()")
        value_before_action = self.portfolio.state.equity
        previous_drawdown = self.portfolio.state.drawdown
        previous_max_drawdown = self.portfolio.state.max_drawdown_depth
        projected = project_action(action, self.config.portfolio, self.config.option)
        rebalance = self.portfolio.rebalance(
            projected.stock_weight,
            projected.futures_exposure,
            self.current_market,
            projected.option_coverages,
        )
        transition = self.market.step()
        self.history.append(
            transition.gas_log_return,
            transition.stock_log_return,
            transition.futures_log_return,
        )
        if self.config.option.enabled:
            accounting = self.portfolio.settle(
                transition,
                realized_stock_volatility=self.history.features().stock_volatility,
            )
        else:
            accounting = self.portfolio.settle(transition)
        self.current_market = transition.current
        self.elapsed_steps += 1
        portfolio_volatility = self.history.portfolio_volatility(
            projected.stock_weight,
            projected.futures_exposure,
        )
        bankrupt = bool(
            not np.isfinite(self.portfolio.state.equity)
            or self.portfolio.state.equity
            <= self.config.portfolio.initial_cash * self.config.portfolio.bankruptcy_fraction
        )
        reward, reward_components = calculate_reward(
            value_before_action,
            self.portfolio.state.equity,
            self.config.portfolio.initial_cash,
            accounting.stock_pnl,
            accounting.futures_pnl,
            accounting.cash_interest,
            accounting.stock_dividend,
            abs(rebalance.stock_trade_value)
            * self.config.portfolio.stock_transaction_cost,
            abs(rebalance.futures_trade_notional)
            * self.config.portfolio.futures_transaction_cost,
            accounting.forced_liquidation_cost,
            rebalance.turnover,
            previous_drawdown,
            self.portfolio.state.drawdown,
            self.portfolio.state.margin_required
            / max(abs(self.portfolio.state.equity), 1e-12),
            self.config.portfolio.annual_cash_rate,
            accounting.margin_call,
            bankrupt,
            self.config.market.trading_days,
            self.elapsed_steps,
            self.config.market.episode_steps,
            self.config.reward,
            option_pnl=accounting.option_pnl,
            option_transaction_cost=rebalance.option_transaction_cost,
            previous_max_drawdown=previous_max_drawdown,
            max_drawdown=self.portfolio.state.max_drawdown_depth,
        )
        reward_components["portfolio_volatility"] = float(portfolio_volatility)
        self.cumulative_reward += reward
        terminated = bool(bankrupt or accounting.margin_call)
        truncated = self.elapsed_steps >= self.config.market.episode_steps
        info = self._info(projected, rebalance, accounting, reward_components)
        info["termination_reason"] = (
            "bankruptcy" if bankrupt else "margin_call" if accounting.margin_call else None
        )
        self.trajectory.append(dict(info))
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str | None:
        if self.current_market is None:
            return None
        info = self._info(None, None, None, None)
        text = (
            f"day={info['day']} value={info['portfolio_value']:,.2f} "
            f"cash={info['cash']:,.2f} pnl={info['pnl']:,.2f} "
            f"leverage={info['leverage']:.3f} "
            f"weights=(stock={info['stock_weight']:.3f}, "
            f"futures={info['futures_exposure']:.3f}, cash={info['cash_weight']:.3f})"
        )
        return text if self.render_mode == "ansi" else None

    def _observation(self) -> np.ndarray:
        assert self.current_market is not None
        return build_observation(
            self.current_market,
            self.portfolio,
            self.history,
            self.config.market,
            self.config.environment,
            self.config.option,
        )

    def _info(
        self,
        action: ProjectedAction | None,
        rebalance: RebalanceResult | None,
        accounting: AccountingResult | None,
        reward_components: dict[str, float] | None,
    ) -> dict[str, Any]:
        assert self.current_market is not None
        state = self.portfolio.state
        assert self.current_market.calendar_date is not None
        next_dividend_date = next_contract_date(
            self.current_market.calendar_date,
            self.config.market.stock_dividend_payment_day,
        )
        stock_weight, futures_exposure, cash_weight = self.portfolio.weights(self.current_market)
        option_value = self.portfolio.option_value()
        option_weight = option_value / max(abs(state.equity), 1e-12)
        option_coverages = self.portfolio.option_coverages()
        leverage = stock_weight + abs(futures_exposure) + option_weight
        transaction_cost = 0.0
        if rebalance is not None:
            transaction_cost += rebalance.transaction_cost
        if accounting is not None:
            transaction_cost += accounting.forced_liquidation_cost
        return {
            "step": self.elapsed_steps,
            "day": self.current_market.day,
            "calendar_date": self.current_market.calendar_date.isoformat()
            if self.current_market.calendar_date else None,
            "delivery_date": self.current_market.delivery_date.isoformat()
            if self.current_market.delivery_date else None,
            "true_regime": self.current_market.regime.label,
            "true_volatility": self.current_market.true_gas_volatility,
            "gas_price": self.current_market.gas_spot,
            "futures_price": self.current_market.futures_price,
            "forward_curve": list(self.current_market.forward_curve),
            "forward_delivery_dates": [
                value.isoformat() for value in self.current_market.forward_delivery_dates
            ],
            "stock_price": self.current_market.stock_price,
            "stock_dividend_per_share": self.current_market.stock_dividend_per_share,
            "is_dividend_payment_date": self.current_market.is_dividend_payment_date,
            "next_dividend_date": next_dividend_date.isoformat(),
            "days_to_dividend": (next_dividend_date - self.current_market.calendar_date).days,
            "portfolio_value": state.equity,
            "cash": state.cash,
            "pnl": state.equity - self.config.portfolio.initial_cash,
            "stock_weight": stock_weight,
            "futures_exposure": futures_exposure,
            "cash_weight": cash_weight,
            "option_enabled": self.config.option.enabled,
            "option_weight": option_weight,
            "option_value": option_value,
            "option_coverages": list(option_coverages),
            "option_positions": [
                {
                    "family": position.family,
                    "tenor_days": position.tenor_days,
                    "strike": position.strike,
                    "expiry_day": position.expiry_day,
                    "units": position.units,
                    "mark_price": position.mark_price,
                    "average_entry_price": position.average_entry_price,
                    "implied_volatility": position.implied_volatility,
                    "delta": position.delta,
                    "theta": position.theta,
                    "market_value": position.market_value,
                }
                for position in state.option_positions
            ],
            "leverage": leverage,
            "drawdown": state.drawdown,
            "max_drawdown": state.max_drawdown_depth,
            "margin_required": state.margin_required,
            "margin_ratio": state.margin_ratio,
            "margin_calls": state.margin_calls,
            "margin_call": bool(accounting.margin_call) if accounting else False,
            "transaction_cost": transaction_cost,
            "cumulative_transaction_cost": state.cumulative_transaction_cost,
            "option_transaction_cost": (
                rebalance.option_transaction_cost if rebalance else 0.0
            ),
            "option_trade_value": rebalance.option_trade_value if rebalance else 0.0,
            "option_premium_paid": rebalance.option_premium_paid if rebalance else 0.0,
            "cumulative_option_premium_paid": state.cumulative_option_premium_paid,
            "option_payoff": accounting.option_payoff if accounting else 0.0,
            "cumulative_option_payoff": state.cumulative_option_payoff,
            "option_pnl": accounting.option_pnl if accounting else 0.0,
            "stock_dividend": accounting.stock_dividend if accounting else 0.0,
            "cumulative_dividends": state.cumulative_dividends,
            "stock_pnl": accounting.stock_pnl if accounting else 0.0,
            "futures_pnl": accounting.futures_pnl if accounting else 0.0,
            "cash_interest": accounting.cash_interest if accounting else 0.0,
            "turnover": rebalance.turnover if rebalance else 0.0,
            "stock_trade_value": rebalance.stock_trade_value if rebalance else 0.0,
            "futures_trade_notional": rebalance.futures_trade_notional if rebalance else 0.0,
            "stock_transaction_cost": (
                abs(rebalance.stock_trade_value)
                * self.config.portfolio.stock_transaction_cost
                if rebalance
                else 0.0
            ),
            "futures_transaction_cost": (
                abs(rebalance.futures_trade_notional)
                * self.config.portfolio.futures_transaction_cost
                if rebalance
                else 0.0
            ),
            "forced_liquidation_notional": (
                accounting.forced_liquidation_notional if accounting else 0.0
            ),
            "forced_liquidation_cost": (
                accounting.forced_liquidation_cost if accounting else 0.0
            ),
            "reward": (reward_components or {}).get("total_reward", 0.0),
            "cumulative_reward": self.cumulative_reward,
            "action_projected": bool(action.was_projected) if action else False,
            "reward_components": reward_components or {},
        }
