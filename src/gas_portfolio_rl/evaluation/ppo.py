from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from gas_portfolio_rl.baselines import ConstantAllocationBaseline
from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.evaluation.evaluator import (
    EvaluationResult,
    evaluate_strategies,
    make_default_strategies,
)
from gas_portfolio_rl.evaluation.policy import NormalizedModelStrategy, load_policy_bundle


def evaluate_ppo_bundle(
    config: AppConfig,
    scenarios: Mapping[str, AppConfig],
    *,
    model_path: str | Path,
    normalization_path: str | Path,
    seeds: Sequence[int] | Mapping[str, Sequence[int]],
    include_baselines: bool = True,
) -> EvaluationResult:
    model, normalizer = load_policy_bundle(
        str(model_path), str(normalization_path), config
    )
    strategies = make_default_strategies(config) if include_baselines else {}
    if include_baselines:
        strategies["minimum_stock_unhedged"] = lambda: ConstantAllocationBaseline(
            config.ppo.minimum_stock_weight,
            0.0,
            "minimum_stock_unhedged",
        )
    strategies["ppo"] = lambda: NormalizedModelStrategy(
        model,
        normalizer,
        ppo_config=config.ppo,
        name="ppo",
        deterministic=True,
    )
    return evaluate_strategies(scenarios, strategies, seeds=seeds)
