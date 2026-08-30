from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from gas_portfolio_rl.baselines.runner import StrategyRun


def episode_metrics(
    run: StrategyRun,
    *,
    initial_value: float,
    trading_days: int,
    annual_cash_rate: float,
    cvar_alpha: float = 0.05,
) -> dict[str, float | str | int | bool]:
    rows = run.trajectory
    values = np.asarray([float(row["portfolio_value"]) for row in rows])
    returns = values[1:] / np.maximum(values[:-1], 1e-12) - 1.0
    periods = len(returns)
    final_value = float(values[-1])
    total_return = final_value / initial_value - 1.0
    annual_return = _annualized_return(total_return, periods, trading_days)
    annual_volatility = _annualized_volatility(returns, trading_days)
    excess = returns - annual_cash_rate / trading_days
    sharpe = _ratio(float(np.mean(excess)) * trading_days, annual_volatility)
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(trading_days))
    sortino = _ratio(float(np.mean(excess)) * trading_days, downside_deviation)
    peaks = np.maximum.accumulate(values)
    drawdowns = values / np.maximum(peaks, 1e-12) - 1.0
    max_drawdown = float(-np.min(drawdowns))
    calmar = _ratio(annual_return, max_drawdown)
    tail_count = max(1, int(math.ceil(len(returns) * cvar_alpha)))
    cvar = max(0.0, float(-np.mean(np.sort(returns)[:tail_count]))) if len(returns) else 0.0
    final = rows[-1]
    reward_components = [
        row.get("reward_components", {}) for row in rows[1:]
    ]

    def component_sum(name: str) -> float:
        return float(sum(float(item.get(name, 0.0)) for item in reward_components))

    fatal_penalty_sum = sum(
        component_sum(name)
        for name in (
            "margin_call_penalty",
            "margin_call_loss_penalty",
            "early_termination_penalty",
            "bankruptcy_penalty",
        )
    )
    futures_exposures = np.asarray(
        [float(row["futures_exposure"]) for row in rows[1:]]
    )
    cash_weights = np.asarray([float(row["cash_weight"]) for row in rows[1:]])
    stock_weights = np.asarray([float(row["stock_weight"]) for row in rows[1:]])
    option_weights = np.asarray([float(row.get("option_weight", 0.0)) for row in rows[1:]])
    option_coverages = np.asarray(
        [row.get("option_coverages", (0.0, 0.0, 0.0)) for row in rows[1:]],
        dtype=np.float64,
    )
    if option_coverages.size == 0:
        option_coverages = np.zeros((0, 3), dtype=np.float64)
    return {
        "strategy": run.strategy_name,
        "steps": periods,
        "terminal_value": final_value,
        "net_pnl": final_value - initial_value,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "cvar": cvar,
        "calmar": calmar,
        "total_reward": run.total_reward,
        "profit_reward_sum": component_sum("profit_reward"),
        "hedge_reward_sum": component_sum("hedge_reward"),
        "downside_penalty_sum": component_sum("downside_penalty"),
        "drawdown_penalty_sum": component_sum("drawdown_penalty"),
        "drawdown_budget_penalty_sum": component_sum("drawdown_budget_penalty"),
        "turnover_penalty_sum": component_sum("turnover_penalty"),
        "margin_utilization_penalty_sum": component_sum(
            "margin_utilization_penalty"
        ),
        "terminal_shortfall_penalty_sum": component_sum(
            "terminal_shortfall_penalty"
        ),
        "terminal_drawdown_penalty_sum": component_sum(
            "terminal_drawdown_penalty"
        ),
        "fatal_penalty_sum": float(fatal_penalty_sum),
        "transaction_cost": float(final["cumulative_transaction_cost"]),
        "turnover": float(sum(float(row["turnover"]) for row in rows)),
        "dividends": float(final["cumulative_dividends"]),
        "margin_calls": int(final["margin_calls"]),
        "bankrupt": final.get("termination_reason") == "bankruptcy",
        "terminated": run.terminated,
        "average_leverage": float(np.mean([float(row["leverage"]) for row in rows[1:]])),
        "average_abs_futures_exposure": float(
            np.mean(np.abs(futures_exposures))
        ),
        "average_futures_exposure": float(np.mean(futures_exposures)),
        "long_futures_fraction": float(np.mean(futures_exposures > 1e-6)),
        "short_futures_fraction": float(np.mean(futures_exposures < -1e-6)),
        "average_cash_weight": float(np.mean(cash_weights)),
        "high_cash_fraction": float(np.mean(cash_weights > 0.80)),
        "average_stock_weight": float(np.mean(stock_weights)),
        "stock_action_std": float(np.std(stock_weights)),
        "futures_action_std": float(np.std(futures_exposures)),
        "average_option_weight": float(np.mean(option_weights)),
        "average_option_coverage": float(np.mean(np.sum(option_coverages, axis=1)))
        if len(option_coverages)
        else 0.0,
        "average_short_put_coverage": float(np.mean(option_coverages[:, 0]))
        if len(option_coverages)
        else 0.0,
        "average_medium_put_coverage": float(np.mean(option_coverages[:, 1]))
        if len(option_coverages)
        else 0.0,
        "average_long_put_coverage": float(np.mean(option_coverages[:, 2]))
        if len(option_coverages)
        else 0.0,
        "option_coverage_std": float(np.std(np.sum(option_coverages, axis=1)))
        if len(option_coverages)
        else 0.0,
        "option_premium_paid": float(final.get("cumulative_option_premium_paid", 0.0)),
        "option_payoff": float(final.get("cumulative_option_payoff", 0.0)),
        "option_pnl": float(sum(float(row.get("option_pnl", 0.0)) for row in rows)),
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    confidence_level: float,
    bootstrap_samples: int,
    seed: int = 12345,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        groups.setdefault((str(row["scenario"]), str(row["strategy"])), []).append(row)
    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    metrics = ("total_return", "net_pnl", "annual_volatility", "sharpe", "sortino",
               "max_drawdown", "cvar", "calmar", "total_reward",
               "transaction_cost", "turnover", "dividends",
               "average_futures_exposure", "average_abs_futures_exposure",
               "long_futures_fraction", "short_futures_fraction")
    metrics = metrics + (
        "profit_reward_sum", "hedge_reward_sum", "downside_penalty_sum",
        "drawdown_penalty_sum", "drawdown_budget_penalty_sum",
        "turnover_penalty_sum", "margin_utilization_penalty_sum",
        "terminal_shortfall_penalty_sum", "terminal_drawdown_penalty_sum",
        "fatal_penalty_sum",
        "average_cash_weight", "high_cash_fraction", "average_stock_weight",
        "stock_action_std", "futures_action_std",
        "average_option_weight", "average_option_coverage",
        "average_short_put_coverage", "average_medium_put_coverage",
        "average_long_put_coverage", "option_coverage_std",
        "option_premium_paid", "option_payoff", "option_pnl",
    )
    alpha = 1.0 - confidence_level
    for (scenario, strategy), rows in sorted(groups.items()):
        output: dict[str, Any] = {
            "scenario": scenario,
            "strategy": strategy,
            "episodes": len(rows),
            "margin_call_rate": float(np.mean([row["margin_calls"] > 0 for row in rows])),
            "bankruptcy_rate": float(np.mean([bool(row["bankrupt"]) for row in rows])),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows])
            output[f"{metric}_mean"] = float(np.mean(values))
            output[f"{metric}_median"] = float(np.median(values))
            indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
            boot_means = np.mean(values[indices], axis=1)
            output[f"{metric}_ci_low"] = float(np.quantile(boot_means, alpha / 2))
            output[f"{metric}_ci_high"] = float(np.quantile(boot_means, 1 - alpha / 2))
        result.append(output)
    return result


def _annualized_return(total_return: float, periods: int, trading_days: int) -> float:
    growth = 1.0 + total_return
    if periods == 0 or growth <= 0:
        return -1.0 if growth <= 0 else 0.0
    return float(growth ** (trading_days / periods) - 1.0)


def _annualized_volatility(returns: np.ndarray, trading_days: int) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * math.sqrt(trading_days))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)
