from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from gas_portfolio_rl.baselines import ConstantAllocationBaseline
from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.evaluation.evaluator import EvaluationResult, StrategyFactory


@dataclass(frozen=True)
class GridAllocation:
    name: str
    stock_weight: float
    futures_exposure: float
    option_coverages: tuple[float, float, float] = (0.0, 0.0, 0.0)


def ppo_feasible_grid(config: AppConfig) -> tuple[GridAllocation, ...]:
    """Static axis grid for the feasible PPO action set.

    The legacy grid contains 15 stock/futures allocations. With options enabled,
    it adds half/full put ladders for every stock/futures pair and full coverage
    in each individual tenor at zero futures exposure, for 54 policies in total.
    """
    stock_levels = (
        config.ppo.minimum_stock_weight,
        0.5 * (config.ppo.minimum_stock_weight + config.ppo.maximum_stock_weight),
        config.ppo.maximum_stock_weight,
    )
    allocations: list[GridAllocation] = []
    for stock_weight in stock_levels:
        futures_limit = min(stock_weight, config.ppo.maximum_futures_exposure)
        for ratio in (-1.0, -0.5, 0.0, 0.5, 1.0):
            futures_exposure = ratio * futures_limit
            name = _allocation_name(stock_weight, futures_exposure)
            allocations.append(GridAllocation(name, stock_weight, futures_exposure))
            if config.option.enabled:
                for total_coverage in (0.5, 1.0):
                    ladder = (total_coverage / 3.0,) * 3
                    allocations.append(GridAllocation(
                        _allocation_name(
                            stock_weight,
                            futures_exposure,
                            option_label=f"ladder{round(100 * total_coverage):03d}",
                        ),
                        stock_weight,
                        futures_exposure,
                        ladder,
                    ))
        if config.option.enabled:
            for label, coverages in (
                ("short100", (1.0, 0.0, 0.0)),
                ("medium100", (0.0, 1.0, 0.0)),
                ("long100", (0.0, 0.0, 1.0)),
            ):
                allocations.append(GridAllocation(
                    _allocation_name(stock_weight, 0.0, option_label=label),
                    stock_weight,
                    0.0,
                    coverages,
                ))
    return tuple(allocations)


def make_ppo_feasible_grid_strategies(
    config: AppConfig,
) -> dict[str, StrategyFactory]:
    strategies: dict[str, StrategyFactory] = {}
    for allocation in ppo_feasible_grid(config):
        strategies[allocation.name] = (
            lambda item=allocation: ConstantAllocationBaseline(
                item.stock_weight,
                item.futures_exposure,
                item.name,
                option_coverages=item.option_coverages,
                rebalance_interval=config.ppo.decision_interval,
            )
        )
    return strategies


def build_reward_gate(
    baseline_result: EvaluationResult,
    grid_result: EvaluationResult,
    config: AppConfig,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Build a transparent reward-alignment and boundary-solution gate."""
    allocation_lookup = {item.name: item for item in ppo_feasible_grid(config)}
    summary_lookup = {
        (str(row["scenario"]), str(row["strategy"])): row
        for row in grid_result.summary
    }
    ranking: list[dict[str, Any]] = []
    base_weight = (1.0 - config.ppo.crisis_validation_weight) / 2.0
    for name, allocation in allocation_lookup.items():
        base = summary_lookup[("base", name)]
        calm = summary_lookup[("calm", name)]
        crisis = summary_lookup[("crisis_stress", name)]
        robust_reward = (
            base_weight * float(base["total_reward_mean"])
            + base_weight * float(calm["total_reward_mean"])
            + config.ppo.crisis_validation_weight
            * float(crisis["total_reward_mean"])
        )
        ranking.append({
            "strategy": name,
            "stock_weight": allocation.stock_weight,
            "futures_exposure": allocation.futures_exposure,
            "short_put_coverage": allocation.option_coverages[0],
            "medium_put_coverage": allocation.option_coverages[1],
            "long_put_coverage": allocation.option_coverages[2],
            "total_option_coverage": sum(allocation.option_coverages),
            "robust_reward": robust_reward,
            "base_reward": float(base["total_reward_mean"]),
            "calm_reward": float(calm["total_reward_mean"]),
            "crisis_reward": float(crisis["total_reward_mean"]),
            "base_return": float(base["total_return_mean"]),
            "calm_return": float(calm["total_return_mean"]),
            "crisis_return": float(crisis["total_return_mean"]),
            "maximum_margin_call_rate": max(
                float(base["margin_call_rate"]),
                float(calm["margin_call_rate"]),
                float(crisis["margin_call_rate"]),
            ),
            "maximum_bankruptcy_rate": max(
                float(base["bankruptcy_rate"]),
                float(calm["bankruptcy_rate"]),
                float(crisis["bankruptcy_rate"]),
            ),
        })
    ranking.sort(key=lambda row: float(row["robust_reward"]), reverse=True)
    best = ranking[0]
    same_stock_unhedged = next(
        row
        for row in ranking
        if float(row["stock_weight"]) == float(best["stock_weight"])
        and abs(float(row["futures_exposure"])) < 1e-12
        and float(row["total_option_coverage"]) < 1e-12
    )
    best_same_stock_hedge = max(
        (
            row
            for row in ranking
            if float(row["stock_weight"]) == float(best["stock_weight"])
            and (
                abs(float(row["futures_exposure"])) >= 1e-12
                or float(row["total_option_coverage"]) >= 1e-12
            )
        ),
        key=lambda row: float(row["crisis_reward"]),
    )
    baseline_alignment = all(
        bool(row["passes_ppo_feasible_alignment"])
        for row in baseline_result.reward_diagnostics
    )
    grid_alignment = all(
        bool(row["passes_ppo_feasible_alignment"])
        for row in grid_result.reward_diagnostics
    )
    checks = {
        "baseline_ppo_feasible_alignment": baseline_alignment,
        "grid_ppo_feasible_alignment": grid_alignment,
        "robust_optimum_has_no_margin_calls": float(best["maximum_margin_call_rate"])
        == 0.0,
        "robust_optimum_has_no_bankruptcies": float(best["maximum_bankruptcy_rate"])
        == 0.0,
    }
    diagnostics = {
        "robust_optimum_has_interior_stock": (
            float(best["stock_weight"]) > config.ppo.minimum_stock_weight + 1e-9
            and float(best["stock_weight"])
            < config.ppo.maximum_stock_weight - 1e-9
        ),
        "robust_optimum_uses_futures": abs(float(best["futures_exposure"]))
        >= config.ppo.futures_action_deadband,
        "crisis_hedge_improves_same_stock_reward": float(
            best_same_stock_hedge["crisis_reward"]
        )
        > float(same_stock_unhedged["crisis_reward"]),
    }
    if config.option.enabled:
        diagnostics["robust_optimum_uses_options"] = (
            float(best["total_option_coverage"])
            >= config.ppo.option_coverage_deadband
        )
    gate = {
        "passes_reward_gate": all(checks.values()),
        "checks": checks,
        "diagnostics": diagnostics,
        "best_grid_policy": best,
        "same_stock_unhedged_policy": same_stock_unhedged,
        "best_same_stock_hedge_policy": best_same_stock_hedge,
        "baseline_diagnostics": list(baseline_result.reward_diagnostics),
        "grid_diagnostics": list(grid_result.reward_diagnostics),
        "grid_policies": len(ranking),
    }
    return gate, tuple(ranking)


def save_reward_gate(
    gate: dict[str, Any],
    ranking: tuple[dict[str, Any], ...],
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "reward_gate.json").open("w", encoding="utf-8") as stream:
        json.dump(gate, stream, indent=2, ensure_ascii=False)
    with (destination / "reward_grid_robust_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)


def _allocation_name(
    stock_weight: float,
    futures_exposure: float,
    *,
    option_label: str | None = None,
) -> str:
    stock = round(100 * stock_weight)
    futures = round(100 * abs(futures_exposure))
    sign = "m" if futures_exposure < 0 else "p"
    name = f"grid_s{stock:03d}_f{sign}{futures:03d}"
    return name if option_label is None else f"{name}_o{option_label}"
