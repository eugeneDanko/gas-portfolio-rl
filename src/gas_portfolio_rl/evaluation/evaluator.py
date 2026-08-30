from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gas_portfolio_rl.baselines import (
    ConstantAllocationBaseline,
    ProtectivePutBaseline,
    RelativeMomentumBaseline,
    RollingHedgeBaseline,
    SignedRollingHedgeBaseline,
    run_strategy,
)
from gas_portfolio_rl.baselines.runner import Strategy
from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv
from gas_portfolio_rl.evaluation.metrics import aggregate_records, episode_metrics

StrategyFactory = Callable[[], Strategy]


@dataclass(frozen=True)
class EvaluationResult:
    records: tuple[dict[str, Any], ...]
    summary: tuple[dict[str, Any], ...]
    paired_deltas: tuple[dict[str, Any], ...]
    reward_diagnostics: tuple[dict[str, Any], ...]


def make_default_strategies(config: AppConfig) -> dict[str, StrategyFactory]:
    evaluation = config.evaluation
    strategies: dict[str, StrategyFactory] = {
        "cash": lambda: ConstantAllocationBaseline(0.0, 0.0, "cash"),
        "stock_only": lambda: ConstantAllocationBaseline(1.0, 0.0, "stock_only"),
        "fixed_hedge": lambda: ConstantAllocationBaseline(
            evaluation.fixed_stock_weight,
            evaluation.fixed_futures_exposure,
            "fixed_hedge",
        ),
        "relative_momentum": lambda: RelativeMomentumBaseline(
            config.baselines.relative_momentum,
            trading_days=config.market.trading_days,
        ),
        "rolling_hedge": lambda: RollingHedgeBaseline(config.baselines.rolling_hedge),
        "signed_rolling_hedge": lambda: SignedRollingHedgeBaseline(
            config.baselines.signed_rolling_hedge
        ),
    }
    if config.option.enabled:
        stock_weight = evaluation.fixed_stock_weight
        strategies.update(
            {
                "protective_put_short": lambda: ProtectivePutBaseline(
                    stock_weight=stock_weight,
                    tenor_allocation=(1.0, 0.0, 0.0),
                    name="protective_put_short",
                ),
                "protective_put_medium": lambda: ProtectivePutBaseline(
                    stock_weight=stock_weight,
                    tenor_allocation=(0.0, 1.0, 0.0),
                    name="protective_put_medium",
                ),
                "protective_put_long": lambda: ProtectivePutBaseline(
                    stock_weight=stock_weight,
                    tenor_allocation=(0.0, 0.0, 1.0),
                    name="protective_put_long",
                ),
                "put_ladder": lambda: ProtectivePutBaseline(
                    stock_weight=stock_weight,
                    tenor_allocation=(1.0, 1.0, 1.0),
                    name="put_ladder",
                ),
                "minimum_stock_put_ladder": lambda: ProtectivePutBaseline(
                    stock_weight=config.ppo.minimum_stock_weight,
                    tenor_allocation=(1.0, 1.0, 1.0),
                    name="minimum_stock_put_ladder",
                ),
            }
        )
    return strategies


def evaluate_strategies(
    scenarios: Mapping[str, AppConfig],
    strategies: Mapping[str, StrategyFactory],
    *,
    seeds: Sequence[int] | Mapping[str, Sequence[int]],
) -> EvaluationResult:
    records: list[dict[str, Any]] = []
    for scenario_name, config in scenarios.items():
        scenario_seeds = seeds[scenario_name] if isinstance(seeds, Mapping) else seeds
        for seed in scenario_seeds:
            for strategy_name, factory in strategies.items():
                run = run_strategy(GasPortfolioEnv(config), factory(), seed=int(seed))
                row = episode_metrics(
                    run,
                    initial_value=config.portfolio.initial_cash,
                    trading_days=config.market.trading_days,
                    annual_cash_rate=config.portfolio.annual_cash_rate,
                    cvar_alpha=config.evaluation.cvar_alpha,
                )
                row.update(scenario=scenario_name, seed=int(seed), strategy=strategy_name)
                records.append(row)
    first_config = next(iter(scenarios.values()))
    summary = aggregate_records(
        records,
        confidence_level=first_config.evaluation.confidence_level,
        bootstrap_samples=first_config.evaluation.bootstrap_samples,
    )
    return EvaluationResult(
        records=tuple(records),
        summary=tuple(summary),
        paired_deltas=tuple(_paired_deltas(records, benchmark="rolling_hedge")),
        reward_diagnostics=tuple(_reward_diagnostics(records)),
    )


def _paired_deltas(
    records: Sequence[Mapping[str, Any]], benchmark: str
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["scenario"]), int(row["seed"]), str(row["strategy"])): row
        for row in records
    }
    groups: dict[tuple[str, str], list[float]] = {}
    for row in records:
        strategy = str(row["strategy"])
        if strategy == benchmark:
            continue
        key = (str(row["scenario"]), int(row["seed"]), benchmark)
        if key not in lookup:
            continue
        delta = float(row["total_return"]) - float(lookup[key]["total_return"])
        groups.setdefault((str(row["scenario"]), strategy), []).append(delta)
    return [
        {
            "scenario": scenario,
            "strategy": strategy,
            "benchmark": benchmark,
            "mean_total_return_delta": float(np.mean(values)),
            "median_total_return_delta": float(np.median(values)),
            "win_rate": float(np.mean(np.asarray(values) > 0)),
        }
        for (scenario, strategy), values in sorted(groups.items())
    ]


def _reward_diagnostics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scenarios = sorted({str(row["scenario"]) for row in records})
    diagnostics: list[dict[str, Any]] = []
    for scenario in scenarios:
        rows = [row for row in records if row["scenario"] == scenario]
        correlation = _reward_utility_correlation(rows)
        feasible_rows = [
            row
            for row in rows
            if str(row["strategy"]) in {
                "fixed_hedge",
                "rolling_hedge",
                "signed_rolling_hedge",
                "protective_put_short",
                "protective_put_medium",
                "protective_put_long",
                "put_ladder",
                "minimum_stock_put_ladder",
                "ppo",
            }
            or str(row["strategy"]).startswith("grid_")
        ]
        feasible_correlation = _reward_utility_correlation(feasible_rows)
        diagnostics.append({
            "scenario": scenario,
            "reward_utility_correlation": correlation,
            "passes_minimum_alignment": correlation >= 0.50,
            "episodes": len(rows),
            "ppo_feasible_reward_utility_correlation": feasible_correlation,
            "passes_ppo_feasible_alignment": feasible_correlation >= 0.50,
            "ppo_feasible_episodes": len(feasible_rows),
        })
    return diagnostics


def _reward_utility_correlation(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    rewards = np.asarray([float(row["total_reward"]) for row in rows])
    utility = np.asarray([
        float(row["total_return"])
        - float(row["max_drawdown"])
        - float(row["cvar"])
        - float(bool(row["bankrupt"])) * 5.0
        - float(int(row["margin_calls"]) > 0) * 2.0
        for row in rows
    ])
    if np.std(rewards) <= 1e-12 or np.std(utility) <= 1e-12:
        return 0.0
    return float(np.corrcoef(rewards, utility)[0, 1])
