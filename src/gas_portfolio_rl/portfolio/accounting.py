from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from gas_portfolio_rl.config.models import OptionConfig, PortfolioConfig
from gas_portfolio_rl.portfolio.options import (
    OptionPosition,
    initialize_option_ladder,
    mark_option,
)
from gas_portfolio_rl.simulation.models import MarketState, MarketTransition


@dataclass(frozen=True)
class PortfolioState:
    cash: float
    stock_shares: float
    futures_contracts: float
    futures_target_notional: float
    equity: float
    peak_equity: float
    drawdown: float
    max_drawdown_depth: float
    margin_required: float
    margin_ratio: float
    cumulative_transaction_cost: float
    cumulative_dividends: float
    margin_calls: int
    option_positions: tuple[OptionPosition, ...]
    cumulative_option_premium_paid: float
    cumulative_option_payoff: float
    option_premium_year: int
    option_premium_paid_this_year: float


@dataclass(frozen=True)
class RebalanceResult:
    transaction_cost: float
    turnover: float
    stock_trade_value: float
    futures_trade_notional: float
    option_trade_value: float = 0.0
    option_transaction_cost: float = 0.0
    option_premium_paid: float = 0.0


@dataclass(frozen=True)
class AccountingResult:
    stock_pnl: float
    futures_pnl: float
    cash_interest: float
    forced_liquidation_cost: float
    margin_call: bool
    stock_dividend: float = 0.0
    forced_liquidation_notional: float = 0.0
    option_pnl: float = 0.0
    option_payoff: float = 0.0


class Portfolio:
    """Cash/stock/futures accounting; futures have zero asset value."""

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        trading_days: int = 252,
        option_config: OptionConfig | None = None,
        annual_stock_dividend_yield: float = 0.0,
    ) -> None:
        self.config = config or PortfolioConfig()
        self.option_config = option_config or OptionConfig()
        self.trading_days = trading_days
        self.annual_stock_dividend_yield = annual_stock_dividend_yield
        self.state = self._initial_state()

    def reset(self, market: MarketState | None = None) -> PortfolioState:
        stock_price = market.stock_price if market is not None else 100.0
        current_day = market.day if market is not None else 0
        positions = self._new_option_ladder(stock_price, current_day)
        self.state = self._initial_state(option_positions=positions)
        return self.state

    def rebalance(
        self,
        target_stock_weight: float,
        target_futures_exposure: float,
        market: MarketState,
        target_option_coverages: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> RebalanceResult:
        if not 0.0 <= target_stock_weight <= 1.0:
            raise ValueError("target_stock_weight must be in [0, 1]")
        if not -1.0 <= target_futures_exposure <= 1.0:
            raise ValueError("target_futures_exposure must be in [-1, 1]")
        if len(target_option_coverages) != 3 or any(
            coverage < 0 for coverage in target_option_coverages
        ):
            raise ValueError("target_option_coverages must contain three non-negative values")
        coverage_sum = sum(target_option_coverages)
        if coverage_sum > self.option_config.maximum_total_coverage + 1e-9:
            raise ValueError("total option coverage exceeds configured maximum")

        value = self.state.equity
        current_stock_value = self.state.stock_shares * market.stock_price
        current_futures_notional = self.state.futures_contracts * market.futures_price
        target_stock_value = target_stock_weight * value
        target_futures_notional = target_futures_exposure * value
        stock_trade = target_stock_value - current_stock_value
        futures_trade = target_futures_notional - current_futures_notional
        base_cost = (
            abs(stock_trade) * self.config.stock_transaction_cost
            + abs(futures_trade) * self.config.futures_transaction_cost
        )
        cash = self.state.cash - stock_trade - base_cost
        shares = target_stock_value / market.stock_price
        contracts = target_futures_notional / market.futures_price
        required = self.config.margin_rate * abs(target_futures_notional)
        option_positions = self.state.option_positions
        option_trade_value = 0.0
        option_cost = 0.0
        premium_paid = 0.0
        premium_year = market.day // self.trading_days
        paid_this_year = self.state.option_premium_paid_this_year
        if premium_year != self.state.option_premium_year:
            paid_this_year = 0.0

        if self.option_config.enabled:
            targets = tuple(coverage * shares for coverage in target_option_coverages)
            option_positions, cash, sell_value, sell_cost = self._execute_option_sells(
                option_positions, targets, cash
            )
            option_trade_value += sell_value
            option_cost += sell_cost
            (
                option_positions,
                cash,
                buy_value,
                buy_cost,
                premium_paid,
            ) = self._execute_option_buys(
                option_positions,
                targets,
                cash,
                value=value,
                margin_required=required,
                premium_paid_this_year=paid_this_year,
            )
            option_trade_value += buy_value
            option_cost += buy_cost
            paid_this_year += premium_paid

        option_value = sum(position.market_value for position in option_positions)
        equity = cash + shares * market.stock_price + option_value
        margin_ratio = _margin_ratio(cash, required)
        cost = base_cost + option_cost
        self.state = replace(
            self.state,
            cash=cash,
            stock_shares=shares,
            futures_contracts=contracts,
            futures_target_notional=target_futures_notional,
            equity=equity,
            margin_required=required,
            margin_ratio=margin_ratio,
            cumulative_transaction_cost=self.state.cumulative_transaction_cost + cost,
            option_positions=option_positions,
            cumulative_option_premium_paid=(
                self.state.cumulative_option_premium_paid + premium_paid
            ),
            option_premium_year=premium_year,
            option_premium_paid_this_year=paid_this_year,
        )
        turnover = (
            abs(stock_trade) + abs(futures_trade) + option_trade_value
        ) / max(value, 1e-12)
        return RebalanceResult(
            cost,
            turnover,
            stock_trade,
            futures_trade,
            option_trade_value,
            option_cost,
            premium_paid,
        )

    def settle(
        self,
        transition: MarketTransition,
        *,
        realized_stock_volatility: float = 0.0,
    ) -> AccountingResult:
        previous = transition.previous
        current = transition.current
        stock_pnl = self.state.stock_shares * (current.stock_price - previous.stock_price)
        futures_pnl = self.state.futures_contracts * (
            transition.futures_settlement_price - previous.futures_price
        )
        cash_interest = (
            self.state.cash * self.config.annual_cash_rate / self.trading_days
        )
        stock_dividend = (
            self.state.stock_shares
            * current.stock_dividend_per_share
        )
        cash = self.state.cash + futures_pnl + cash_interest + stock_dividend

        option_pnl = 0.0
        option_payoff = 0.0
        option_positions: list[OptionPosition] = []
        for position in self.state.option_positions:
            previous_value = position.market_value
            if current.day >= position.expiry_day:
                payoff = position.units * max(position.strike - current.stock_price, 0.0)
                cash += payoff
                option_payoff += payoff
                option_pnl += payoff - previous_value
                replacement = OptionPosition(
                    family=position.family,
                    tenor_days=position.tenor_days,
                    strike=self.option_config.strike_moneyness * current.stock_price,
                    expiry_day=current.day + position.tenor_days,
                )
                option_positions.append(
                    self._mark_option(replacement, current, realized_stock_volatility)
                )
            else:
                marked = self._mark_option(position, current, realized_stock_volatility)
                option_pnl += marked.market_value - previous_value
                option_positions.append(marked)

        contracts = self.state.futures_contracts
        if transition.rolled:
            contracts = self.state.futures_target_notional / current.futures_price

        stock_value = self.state.stock_shares * current.stock_price
        option_value = sum(position.market_value for position in option_positions)
        equity = cash + stock_value + option_value
        current_notional = contracts * current.futures_price
        required = self.config.margin_rate * abs(current_notional)
        margin_call = cash + 1e-9 < required
        forced_cost = 0.0
        forced_notional = 0.0

        if margin_call and abs(current_notional) > 0.0:
            old_abs_notional = abs(current_notional)
            forced_notional = old_abs_notional
            fee = self.config.futures_transaction_cost
            forced_cost = fee * old_abs_notional
            cash -= forced_cost
            contracts = 0.0
            current_notional = 0.0
            required = 0.0
            equity = cash + stock_value + option_value

        peak = max(self.state.peak_equity, equity)
        drawdown = equity / peak - 1.0 if peak > 0 else -1.0
        max_drawdown_depth = max(self.state.max_drawdown_depth, max(-drawdown, 0.0))
        self.state = replace(
            self.state,
            cash=cash,
            futures_contracts=contracts,
            futures_target_notional=current_notional,
            equity=equity,
            peak_equity=peak,
            drawdown=drawdown,
            max_drawdown_depth=max_drawdown_depth,
            margin_required=required,
            margin_ratio=_margin_ratio(cash, required),
            cumulative_transaction_cost=self.state.cumulative_transaction_cost + forced_cost,
            cumulative_dividends=self.state.cumulative_dividends + stock_dividend,
            margin_calls=self.state.margin_calls + int(margin_call),
            option_positions=tuple(option_positions),
            cumulative_option_payoff=self.state.cumulative_option_payoff + option_payoff,
        )
        return AccountingResult(
            stock_pnl=stock_pnl,
            futures_pnl=futures_pnl,
            cash_interest=cash_interest,
            forced_liquidation_cost=forced_cost,
            margin_call=margin_call,
            stock_dividend=stock_dividend,
            forced_liquidation_notional=forced_notional,
            option_pnl=option_pnl,
            option_payoff=option_payoff,
        )

    def stock_value(self, market: MarketState) -> float:
        return self.state.stock_shares * market.stock_price

    def futures_notional(self, market: MarketState) -> float:
        return self.state.futures_contracts * market.futures_price

    def option_value(self) -> float:
        return float(sum(position.market_value for position in self.state.option_positions))

    def option_coverages(self) -> tuple[float, float, float]:
        return tuple(
            position.coverage(self.state.stock_shares)
            for position in self.state.option_positions
        )  # type: ignore[return-value]

    def weights(self, market: MarketState) -> tuple[float, float, float]:
        equity = max(abs(self.state.equity), 1e-12)
        return (
            self.stock_value(market) / equity,
            self.futures_notional(market) / equity,
            self.state.cash / equity,
        )

    def _initial_state(
        self,
        *,
        option_positions: tuple[OptionPosition, ...] = (),
    ) -> PortfolioState:
        value = self.config.initial_cash
        return PortfolioState(
            cash=value,
            stock_shares=0.0,
            futures_contracts=0.0,
            futures_target_notional=0.0,
            equity=value,
            peak_equity=value,
            drawdown=0.0,
            max_drawdown_depth=0.0,
            margin_required=0.0,
            margin_ratio=10.0,
            cumulative_transaction_cost=0.0,
            cumulative_dividends=0.0,
            margin_calls=0,
            option_positions=option_positions,
            cumulative_option_premium_paid=0.0,
            cumulative_option_payoff=0.0,
            option_premium_year=0,
            option_premium_paid_this_year=0.0,
        )

    def _new_option_ladder(
        self, stock_price: float, current_day: int
    ) -> tuple[OptionPosition, ...]:
        if not self.option_config.enabled:
            return ()
        return initialize_option_ladder(
            stock_price,
            current_day,
            self.option_config,
            trading_days=self.trading_days,
            risk_free_rate=self.config.annual_cash_rate,
            dividend_yield=self.annual_stock_dividend_yield,
        )

    def _mark_option(
        self,
        position: OptionPosition,
        market: MarketState,
        realized_volatility: float,
    ) -> OptionPosition:
        return mark_option(
            position,
            market.stock_price,
            market.day,
            self.option_config,
            trading_days=self.trading_days,
            risk_free_rate=self.config.annual_cash_rate,
            dividend_yield=self.annual_stock_dividend_yield,
            realized_volatility=realized_volatility,
        )

    def _execute_option_sells(
        self,
        positions: tuple[OptionPosition, ...],
        targets: tuple[float, float, float],
        cash: float,
    ) -> tuple[tuple[OptionPosition, ...], float, float, float]:
        updated: list[OptionPosition] = []
        traded_value = 0.0
        total_cost = 0.0
        for position, target in zip(positions, targets, strict=True):
            sell_units = max(position.units - target, 0.0)
            if sell_units <= 1e-12:
                updated.append(position)
                continue
            bid = position.mark_price * (1.0 - self.option_config.relative_half_spread)
            commission = (
                sell_units / self.option_config.contract_multiplier
                * self.option_config.commission_per_contract
            )
            cash += sell_units * bid - commission
            traded_value += sell_units * position.mark_price
            total_cost += sell_units * position.mark_price * self.option_config.relative_half_spread + commission
            remaining = max(position.units - sell_units, 0.0)
            updated.append(
                replace(
                    position,
                    units=remaining,
                    average_entry_price=(position.average_entry_price if remaining > 0 else 0.0),
                )
            )
        return tuple(updated), cash, traded_value, total_cost

    def _execute_option_buys(
        self,
        positions: tuple[OptionPosition, ...],
        targets: tuple[float, float, float],
        cash: float,
        *,
        value: float,
        margin_required: float,
        premium_paid_this_year: float,
    ) -> tuple[tuple[OptionPosition, ...], float, float, float, float]:
        desired_units = [max(target - position.units, 0.0) for position, target in zip(positions, targets, strict=True)]
        desired_cash = 0.0
        current_option_value = sum(position.market_value for position in positions)
        for position, units in zip(positions, desired_units, strict=True):
            ask = position.mark_price * (1.0 + self.option_config.relative_half_spread)
            desired_cash += units * ask + (
                units / self.option_config.contract_multiplier
                * self.option_config.commission_per_contract
            )
        if desired_cash <= 1e-12:
            return positions, cash, 0.0, 0.0, 0.0

        cash_capacity = max(
            cash - margin_required - self.config.min_cash_buffer * max(value, 0.0),
            0.0,
        )
        value_capacity = max(
            self.option_config.maximum_option_value_fraction * value - current_option_value,
            0.0,
        )
        annual_capacity = max(
            self.option_config.maximum_annual_premium_fraction * value
            - premium_paid_this_year,
            0.0,
        )
        scale = min(1.0, cash_capacity / desired_cash, annual_capacity / desired_cash)
        mid_value_requested = sum(
            units * position.mark_price
            for position, units in zip(positions, desired_units, strict=True)
        )
        if mid_value_requested > 1e-12:
            scale = min(scale, value_capacity / mid_value_requested)
        scale = float(np.clip(scale, 0.0, 1.0))

        updated: list[OptionPosition] = []
        traded_value = 0.0
        total_cost = 0.0
        premium_paid = 0.0
        for position, desired in zip(positions, desired_units, strict=True):
            units = desired * scale
            if units <= 1e-12:
                updated.append(position)
                continue
            ask = position.mark_price * (1.0 + self.option_config.relative_half_spread)
            commission = (
                units / self.option_config.contract_multiplier
                * self.option_config.commission_per_contract
            )
            paid = units * ask + commission
            cash -= paid
            premium_paid += paid
            traded_value += units * position.mark_price
            total_cost += units * position.mark_price * self.option_config.relative_half_spread + commission
            new_units = position.units + units
            average_entry = (
                position.average_entry_price * position.units + ask * units
            ) / max(new_units, 1e-12)
            updated.append(
                replace(position, units=new_units, average_entry_price=average_entry)
            )
        return tuple(updated), cash, traded_value, total_cost, premium_paid


def _margin_ratio(cash: float, required: float) -> float:
    if required <= 1e-12:
        return 10.0 if cash >= 0.0 else 0.0
    return float(min(max(cash / required, 0.0), 10.0))
