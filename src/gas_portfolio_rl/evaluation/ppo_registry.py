from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.evaluation.evaluator import EvaluationResult


REGISTRY_FIELDS = (
    "run_id",
    "experiment_tag",
    "schema_version",
    "training_seed",
    "total_timesteps",
    "evaluation_split",
    "scenario",
    "episodes",
    "passes_all",
    "initial_value",
    "total_return_mean",
    "total_return_ci_low",
    "total_return_ci_high",
    "net_pnl_mean",
    "annual_volatility_mean",
    "sharpe_mean",
    "sortino_mean",
    "max_drawdown_mean",
    "max_drawdown_ci_low",
    "max_drawdown_ci_high",
    "cvar_mean",
    "calmar_mean",
    "total_reward_mean",
    "total_reward_ci_low",
    "total_reward_ci_high",
    "turnover_mean",
    "transaction_cost_mean",
    "dividends_mean",
    "average_cash_weight_mean",
    "high_cash_fraction_mean",
    "average_stock_weight_mean",
    "average_abs_futures_exposure_mean",
    "average_option_coverage_mean",
    "average_short_put_coverage_mean",
    "average_medium_put_coverage_mean",
    "average_long_put_coverage_mean",
    "option_premium_paid_mean",
    "option_payoff_mean",
    "option_pnl_mean",
    "profit_reward_sum_mean",
    "hedge_reward_sum_mean",
    "downside_penalty_sum_mean",
    "drawdown_penalty_sum_mean",
    "drawdown_budget_penalty_sum_mean",
    "turnover_penalty_sum_mean",
    "margin_utilization_penalty_sum_mean",
    "terminal_shortfall_penalty_sum_mean",
    "terminal_drawdown_penalty_sum_mean",
    "fatal_penalty_sum_mean",
    "margin_call_rate",
    "bankruptcy_rate",
)


def load_ppo_registry(path: str | Path) -> list[dict[str, str]]:
    registry = Path(path)
    if not registry.is_file():
        return []
    with registry.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_ppo_leaderboard(
    registry: Iterable[Mapping[str, Any]],
    *,
    evaluation_split: str = "validation",
) -> list[dict[str, Any]]:
    """Rank complete PPO runs by acceptance, tail reward and crisis protection."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in registry:
        if str(row.get("evaluation_split", "validation")) != evaluation_split:
            continue
        key = (str(row.get("experiment_tag", "unknown")), str(row["run_id"]))
        groups.setdefault(key, []).append(row)

    required = {"base", "calm", "crisis_stress"}
    leaderboard: list[dict[str, Any]] = []
    for (tag, run_id), rows in groups.items():
        by_scenario = {str(row["scenario"]): row for row in rows}
        if not required.issubset(by_scenario):
            continue
        crisis = by_scenario["crisis_stress"]
        rewards = [float(by_scenario[name]["total_reward_mean"]) for name in required]
        returns = [float(by_scenario[name]["total_return_mean"]) for name in required]
        drawdowns = [float(by_scenario[name]["max_drawdown_mean"]) for name in required]
        raw_pass = str(crisis.get("passes_all", "false")).lower()
        leaderboard.append(
            {
                "rank": 0,
                "experiment_tag": tag,
                "run_id": run_id,
                "passes_all": raw_pass in {"true", "1"},
                "worst_scenario_reward": min(rewards),
                "worst_scenario_return": min(returns),
                "worst_max_drawdown": max(drawdowns),
                "crisis_return": float(crisis["total_return_mean"]),
                "crisis_max_drawdown": float(crisis["max_drawdown_mean"]),
                "mean_reward": float(np.mean(rewards)),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            bool(row["passes_all"]),
            float(row["worst_scenario_reward"]),
            -float(row["crisis_max_drawdown"]),
            float(row["worst_scenario_return"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(leaderboard, start=1):
        row["rank"] = index
    return leaderboard


def register_ppo_evaluation(
    result: EvaluationResult,
    *,
    run_dir: str | Path,
    config: AppConfig,
    acceptance: Mapping[str, Any],
    registry_path: str | Path,
    evaluation_split: str = "validation",
) -> list[dict[str, str]]:
    """Upsert scenario-level PPO statistics for cross-version comparisons."""
    run_root = Path(run_dir)
    metadata_path = run_root / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    training_seed = int(metadata.get("training_seed", -1))
    initial_value = float(config.portfolio.initial_cash)
    additions: list[dict[str, Any]] = []
    for row in result.summary:
        if str(row["strategy"]) != "ppo":
            continue
        total_return = float(row["total_return_mean"])
        additions.append(
            {
                "run_id": run_root.name,
                "experiment_tag": config.ppo.experiment_tag,
                "schema_version": config.ppo.version,
                "training_seed": training_seed,
                "total_timesteps": config.ppo.total_timesteps,
                "evaluation_split": evaluation_split,
                "scenario": str(row["scenario"]),
                "episodes": int(row["episodes"]),
                "passes_all": bool(acceptance.get("passes_all", False)),
                "initial_value": initial_value,
                "total_return_mean": total_return,
                "total_return_ci_low": float(row["total_return_ci_low"]),
                "total_return_ci_high": float(row["total_return_ci_high"]),
                "net_pnl_mean": float(row.get("net_pnl_mean", total_return * initial_value)),
                "annual_volatility_mean": float(row.get("annual_volatility_mean", 0.0)),
                "sharpe_mean": float(row["sharpe_mean"]),
                "sortino_mean": float(row.get("sortino_mean", 0.0)),
                "max_drawdown_mean": float(row["max_drawdown_mean"]),
                "max_drawdown_ci_low": float(row["max_drawdown_ci_low"]),
                "max_drawdown_ci_high": float(row["max_drawdown_ci_high"]),
                "cvar_mean": float(row["cvar_mean"]),
                "calmar_mean": float(row.get("calmar_mean", 0.0)),
                "total_reward_mean": float(row["total_reward_mean"]),
                "total_reward_ci_low": float(row["total_reward_ci_low"]),
                "total_reward_ci_high": float(row["total_reward_ci_high"]),
                "turnover_mean": float(row["turnover_mean"]),
                "transaction_cost_mean": float(row.get("transaction_cost_mean", 0.0)),
                "dividends_mean": float(row.get("dividends_mean", 0.0)),
                "average_cash_weight_mean": float(row.get("average_cash_weight_mean", 0.0)),
                "high_cash_fraction_mean": float(row.get("high_cash_fraction_mean", 0.0)),
                "average_stock_weight_mean": float(row["average_stock_weight_mean"]),
                "average_abs_futures_exposure_mean": float(
                    row["average_abs_futures_exposure_mean"]
                ),
                "average_option_coverage_mean": float(
                    row.get("average_option_coverage_mean", 0.0)
                ),
                "average_short_put_coverage_mean": float(row.get("average_short_put_coverage_mean", 0.0)),
                "average_medium_put_coverage_mean": float(row.get("average_medium_put_coverage_mean", 0.0)),
                "average_long_put_coverage_mean": float(row.get("average_long_put_coverage_mean", 0.0)),
                "option_premium_paid_mean": float(row.get("option_premium_paid_mean", 0.0)),
                "option_payoff_mean": float(row.get("option_payoff_mean", 0.0)),
                "option_pnl_mean": float(row.get("option_pnl_mean", 0.0)),
                "profit_reward_sum_mean": float(row.get("profit_reward_sum_mean", 0.0)),
                "hedge_reward_sum_mean": float(row.get("hedge_reward_sum_mean", 0.0)),
                "downside_penalty_sum_mean": float(row.get("downside_penalty_sum_mean", 0.0)),
                "drawdown_penalty_sum_mean": float(row.get("drawdown_penalty_sum_mean", 0.0)),
                "drawdown_budget_penalty_sum_mean": float(row.get("drawdown_budget_penalty_sum_mean", 0.0)),
                "turnover_penalty_sum_mean": float(row.get("turnover_penalty_sum_mean", 0.0)),
                "margin_utilization_penalty_sum_mean": float(row.get("margin_utilization_penalty_sum_mean", 0.0)),
                "terminal_shortfall_penalty_sum_mean": float(row.get("terminal_shortfall_penalty_sum_mean", 0.0)),
                "terminal_drawdown_penalty_sum_mean": float(row.get("terminal_drawdown_penalty_sum_mean", 0.0)),
                "fatal_penalty_sum_mean": float(row.get("fatal_penalty_sum_mean", 0.0)),
                "margin_call_rate": float(row["margin_call_rate"]),
                "bankruptcy_rate": float(row["bankruptcy_rate"]),
            }
        )

    registry = Path(registry_path)
    existing = load_ppo_registry(registry)
    keys = {
        (str(row["run_id"]), str(row["evaluation_split"]), str(row["scenario"]))
        for row in additions
    }
    retained = [
        row
        for row in existing
        if (row["run_id"], row["evaluation_split"], row["scenario"]) not in keys
    ]
    materialized: list[Mapping[str, Any]] = [*retained, *additions]
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(materialized)
    return load_ppo_registry(registry)
