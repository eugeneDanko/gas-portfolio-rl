from __future__ import annotations

from typing import Any

from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.evaluation.evaluator import EvaluationResult


def assess_ppo(result: EvaluationResult, config: AppConfig) -> dict[str, Any]:
    lookup = {
        (str(row["scenario"]), str(row["strategy"])): row
        for row in result.summary
    }

    def value(scenario: str, strategy: str, metric: str) -> float:
        return float(lookup[(scenario, strategy)][metric])

    required = [
        ("base", "ppo"),
        ("base", "cash"),
        ("calm", "ppo"),
        ("calm", "cash"),
        ("crisis_stress", "ppo"),
        ("crisis_stress", "cash"),
        ("crisis_stress", "signed_rolling_hedge"),
    ]
    required.extend(
        (scenario, "minimum_stock_unhedged")
        for scenario in ("base", "calm", "crisis_stress")
    )
    if config.option.enabled:
        required.extend(
            (scenario, "minimum_stock_put_ladder")
            for scenario in ("base", "calm", "crisis_stress")
        )
    missing = [key for key in required if key not in lookup]
    if missing:
        raise ValueError(f"Missing PPO acceptance rows: {missing}")

    ppo_rows = [row for row in result.summary if row["strategy"] == "ppo"]
    normal_weight = (1.0 - config.ppo.crisis_validation_weight) / 2.0

    def robust_reward(strategy: str) -> float:
        return (
            normal_weight * value("base", strategy, "total_reward_mean")
            + normal_weight * value("calm", strategy, "total_reward_mean")
            + config.ppo.crisis_validation_weight
            * value("crisis_stress", strategy, "total_reward_mean")
        )

    checks = {
        "zero_bankruptcy": max(float(row["bankruptcy_rate"]) for row in ppo_rows) == 0.0,
        "margin_call_rate_at_most_1pct": max(
            float(row["margin_call_rate"]) for row in ppo_rows
        ) <= 0.01,
        "base_reward_above_cash": value("base", "ppo", "total_reward_mean")
        > value("base", "cash", "total_reward_mean"),
        "base_return_above_cash": value("base", "ppo", "total_return_mean")
        > value("base", "cash", "total_return_mean"),
        "calm_reward_above_cash": value("calm", "ppo", "total_reward_mean")
        > value("calm", "cash", "total_reward_mean"),
        "calm_return_above_cash": value("calm", "ppo", "total_return_mean")
        > value("calm", "cash", "total_return_mean"),
        "crisis_reward_above_signed_hedge": value(
            "crisis_stress", "ppo", "total_reward_mean"
        ) > value("crisis_stress", "signed_rolling_hedge", "total_reward_mean"),
        "crisis_drawdown_no_worse_than_signed_hedge": value(
            "crisis_stress", "ppo", "max_drawdown_mean"
        ) <= value("crisis_stress", "signed_rolling_hedge", "max_drawdown_mean"),
        "crisis_reward_above_cash": value(
            "crisis_stress", "ppo", "total_reward_mean"
        ) > value("crisis_stress", "cash", "total_reward_mean"),
        "not_constant_cash": value("base", "ppo", "high_cash_fraction_mean")
        <= config.ppo.maximum_high_cash_fraction,
        "minimum_stock_participation": value(
            "base", "ppo", "average_stock_weight_mean"
        ) >= config.ppo.minimum_average_stock_weight,
        "turnover_within_limit": max(
            value(str(row["scenario"]), "ppo", "turnover_mean") for row in ppo_rows
        ) <= config.ppo.maximum_turnover,
        "robust_reward_above_minimum_stock_unhedged": robust_reward("ppo")
        > robust_reward("minimum_stock_unhedged"),
    }
    if config.option.enabled:
        checks.update(
            {
                "robust_reward_above_minimum_stock_put_ladder": robust_reward("ppo")
                > robust_reward("minimum_stock_put_ladder"),
                "crisis_drawdown_no_worse_than_minimum_stock_put_ladder": value(
                    "crisis_stress", "ppo", "max_drawdown_mean"
                )
                <= value(
                    "crisis_stress", "minimum_stock_put_ladder", "max_drawdown_mean"
                ),
            }
        )
    observed = {
        "maximum_margin_call_rate": max(
            float(row["margin_call_rate"]) for row in ppo_rows
        ),
        "maximum_bankruptcy_rate": max(
            float(row["bankruptcy_rate"]) for row in ppo_rows
        ),
        "base_ppo_reward": value("base", "ppo", "total_reward_mean"),
        "base_cash_reward": value("base", "cash", "total_reward_mean"),
        "base_ppo_return": value("base", "ppo", "total_return_mean"),
        "base_cash_return": value("base", "cash", "total_return_mean"),
        "calm_ppo_return": value("calm", "ppo", "total_return_mean"),
        "calm_cash_return": value("calm", "cash", "total_return_mean"),
        "base_high_cash_fraction": value("base", "ppo", "high_cash_fraction_mean"),
        "crisis_ppo_reward": value("crisis_stress", "ppo", "total_reward_mean"),
        "crisis_signed_reward": value(
            "crisis_stress", "signed_rolling_hedge", "total_reward_mean"
        ),
        "crisis_cash_reward": value("crisis_stress", "cash", "total_reward_mean"),
        "crisis_ppo_drawdown": value("crisis_stress", "ppo", "max_drawdown_mean"),
        "crisis_signed_drawdown": value(
            "crisis_stress", "signed_rolling_hedge", "max_drawdown_mean"
        ),
        "base_average_stock_weight": value(
            "base", "ppo", "average_stock_weight_mean"
        ),
        "base_average_option_coverage": value(
            "base", "ppo", "average_option_coverage_mean"
        )
        if config.option.enabled
        else 0.0,
        "maximum_turnover": max(
            value(str(row["scenario"]), "ppo", "turnover_mean") for row in ppo_rows
        ),
        "ppo_robust_reward": robust_reward("ppo"),
        "minimum_stock_unhedged_robust_reward": robust_reward(
            "minimum_stock_unhedged"
        ),
    }
    if config.option.enabled:
        observed.update(
            {
                "minimum_stock_put_ladder_robust_reward": robust_reward(
                    "minimum_stock_put_ladder"
                ),
                "crisis_minimum_stock_put_ladder_drawdown": value(
                    "crisis_stress", "minimum_stock_put_ladder", "max_drawdown_mean"
                ),
            }
        )
    return {
        "passes_all": all(checks.values()),
        "checks": checks,
        "observed": observed,
    }
