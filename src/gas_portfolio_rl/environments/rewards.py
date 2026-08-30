from __future__ import annotations

import math

from gas_portfolio_rl.config.models import RewardConfig


def calculate_reward(
    value_before_action: float,
    value_after_step: float,
    initial_value: float,
    stock_pnl: float,
    futures_pnl: float,
    cash_interest: float,
    stock_dividend: float,
    stock_transaction_cost: float,
    futures_transaction_cost: float,
    forced_liquidation_cost: float,
    turnover: float,
    previous_drawdown: float,
    drawdown: float,
    margin_utilization: float,
    annual_cash_rate: float,
    margin_call: bool,
    bankrupt: bool,
    trading_days: int,
    current_step: int,
    episode_steps: int,
    config: RewardConfig,
    option_pnl: float = 0.0,
    option_transaction_cost: float = 0.0,
    previous_max_drawdown: float | None = None,
    max_drawdown: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Capital-preservation reward with an explicit hedge-effectiveness signal.

    Profit enters exactly once through excess_return. The P&L decomposition is
    retained for diagnostics and for estimating the counterfactual return of
    the same stock/cash portfolio without its futures leg.
    """
    if initial_value <= 0 or trading_days <= 0 or episode_steps <= 0:
        raise ValueError("initial_value, trading_days and episode_steps must be positive")
    if current_step < 0 or annual_cash_rate < 0:
        raise ValueError("current_step and annual_cash_rate must be non-negative")

    safe_before = max(value_before_action, 1e-12)
    finite_after = value_after_step if math.isfinite(value_after_step) else 0.0
    net_profit = finite_after - value_before_action
    net_return = net_profit / safe_before
    cash_benchmark_return = annual_cash_rate / trading_days
    excess_return = net_return - cash_benchmark_return
    utility_return = _asymmetric_utility(excess_return, config.loss_aversion)
    profit_reward = config.profit_weight * utility_return

    futures_net_pnl = futures_pnl - futures_transaction_cost - forced_liquidation_cost
    futures_return_contribution = futures_net_pnl / safe_before
    option_net_pnl = option_pnl - option_transaction_cost
    option_return_contribution = option_net_pnl / safe_before
    hedge_return_contribution = futures_return_contribution + option_return_contribution
    unhedged_excess_return = excess_return - hedge_return_contribution
    unhedged_utility_return = _asymmetric_utility(
        unhedged_excess_return, config.loss_aversion
    )
    raw_hedge_effectiveness = utility_return - unhedged_utility_return
    clipped_hedge_effectiveness = min(
        max(raw_hedge_effectiveness, -config.hedge_effectiveness_clip),
        config.hedge_effectiveness_clip,
    )
    hedge_reward = config.hedge_effectiveness_weight * clipped_hedge_effectiveness

    actual_downside = max(-excess_return, 0.0)
    downside_cost = config.downside_penalty * actual_downside**2
    prior_drawdown_depth = max(-previous_drawdown, 0.0)
    drawdown_depth = max(-drawdown, 0.0)
    prior_max_depth = (
        prior_drawdown_depth
        if previous_max_drawdown is None
        else max(previous_max_drawdown, 0.0)
    )
    max_depth = (
        max(prior_max_depth, drawdown_depth)
        if max_drawdown is None
        else max(max_drawdown, drawdown_depth, prior_max_depth)
    )
    drawdown_increase = max(max_depth - prior_max_depth, 0.0)
    drawdown_cost = config.drawdown_increase_penalty * drawdown_increase
    drawdown_budget_excess = max(drawdown_depth - config.maximum_drawdown, 0.0)
    drawdown_budget_cost = (
        config.drawdown_budget_penalty * drawdown_budget_excess**2
    )
    turnover_cost = config.turnover_penalty * max(turnover, 0.0)
    margin_excess = max(margin_utilization - config.safe_margin_utilization, 0.0)
    margin_cost = config.margin_utilization_penalty * margin_excess**2

    episode_complete = current_step >= episode_steps and not (margin_call or bankrupt)
    cash_terminal_value = initial_value * (
        1.0 + annual_cash_rate / trading_days
    ) ** min(current_step, episode_steps)
    shortfall_fraction = max(cash_terminal_value - finite_after, 0.0) / initial_value
    terminal_shortfall_cost = (
        config.terminal_shortfall_penalty * shortfall_fraction * float(episode_complete)
    )
    terminal_drawdown_excess = max(max_depth - config.maximum_drawdown, 0.0)
    terminal_drawdown_cost = (
        config.terminal_drawdown_penalty
        * terminal_drawdown_excess
        * float(episode_complete)
    )

    margin_call_cost = config.margin_call_penalty * float(margin_call)
    loss_fraction = max(0.0, (initial_value - finite_after) / initial_value)
    margin_loss_cost = config.margin_call_loss_penalty * loss_fraction * float(margin_call)
    remaining_fraction = max(episode_steps - current_step, 0) / episode_steps
    early_termination_cost = (
        config.early_termination_penalty
        * remaining_fraction
        * float(margin_call or bankrupt)
    )
    bankruptcy_cost = float(bankrupt) * (
        config.bankruptcy_penalty + config.bankruptcy_loss_penalty * loss_fraction
    )

    reward = (
        profit_reward
        + hedge_reward
        - downside_cost
        - drawdown_cost
        - drawdown_budget_cost
        - turnover_cost
        - margin_cost
        - terminal_shortfall_cost
        - terminal_drawdown_cost
        - margin_call_cost
        - margin_loss_cost
        - early_termination_cost
        - bankruptcy_cost
    )
    decomposed_profit = (
        stock_pnl
        + futures_pnl
        + option_pnl
        + cash_interest
        + stock_dividend
        - stock_transaction_cost
        - futures_transaction_cost
        - option_transaction_cost
        - forced_liquidation_cost
    )
    return float(reward), {
        "net_profit": float(net_profit),
        "decomposed_profit": float(decomposed_profit),
        "pnl_decomposition_error": float(net_profit - decomposed_profit),
        "net_return": float(net_return),
        "cash_benchmark_return": float(cash_benchmark_return),
        "excess_return": float(excess_return),
        "profit_reward": float(profit_reward),
        "futures_return_contribution": float(futures_return_contribution),
        "option_return_contribution": float(option_return_contribution),
        "hedge_return_contribution": float(hedge_return_contribution),
        "unhedged_excess_return": float(unhedged_excess_return),
        "unhedged_utility_return": float(unhedged_utility_return),
        "hedge_effectiveness": float(clipped_hedge_effectiveness),
        "hedge_reward": float(hedge_reward),
        "downside_penalty": float(downside_cost),
        "drawdown_increase": float(drawdown_increase),
        "drawdown_penalty": float(drawdown_cost),
        "max_drawdown": float(max_depth),
        "drawdown_budget_excess": float(drawdown_budget_excess),
        "drawdown_budget_penalty": float(drawdown_budget_cost),
        "turnover_penalty": float(turnover_cost),
        "margin_utilization": float(margin_utilization),
        "margin_utilization_penalty": float(margin_cost),
        "terminal_shortfall_penalty": float(terminal_shortfall_cost),
        "terminal_drawdown_penalty": float(terminal_drawdown_cost),
        "margin_call_penalty": float(margin_call_cost),
        "margin_call_loss_penalty": float(margin_loss_cost),
        "early_termination_penalty": float(early_termination_cost),
        "bankruptcy_penalty": float(bankruptcy_cost),
        "total_reward": float(reward),
    }


def _asymmetric_utility(value: float, loss_aversion: float) -> float:
    return value if value >= 0.0 else loss_aversion * value
