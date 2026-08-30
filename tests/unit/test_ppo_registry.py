from __future__ import annotations

from dataclasses import replace

from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.evaluation.evaluator import EvaluationResult
from gas_portfolio_rl.evaluation.ppo_registry import (
    build_ppo_leaderboard,
    register_ppo_evaluation,
)
from gas_portfolio_rl.visualization import plot_versioned_pnl_comparison


def _ppo_summary(scenario: str, total_return: float) -> dict[str, float | int | str]:
    return {
        "scenario": scenario,
        "strategy": "ppo",
        "episodes": 5,
        "total_return_mean": total_return,
        "total_return_ci_low": total_return - 0.01,
        "total_return_ci_high": total_return + 0.01,
        "net_pnl_mean": total_return * 1_000_000,
        "sharpe_mean": 0.5,
        "max_drawdown_mean": 0.1,
        "max_drawdown_ci_low": 0.09,
        "max_drawdown_ci_high": 0.11,
        "cvar_mean": 0.02,
        "total_reward_mean": 0.1,
        "total_reward_ci_low": 0.0,
        "total_reward_ci_high": 0.2,
        "turnover_mean": 2.0,
        "average_stock_weight_mean": 0.4,
        "average_abs_futures_exposure_mean": 0.2,
        "average_option_coverage_mean": 0.3,
        "margin_call_rate": 0.0,
        "bankruptcy_rate": 0.0,
    }


def test_registry_upserts_ppo_and_supports_versioned_pnl_plot(tmp_path) -> None:
    scenarios = ("base", "calm", "crisis_stress")
    result = EvaluationResult(
        records=(),
        summary=tuple(_ppo_summary(name, 0.05) for name in scenarios),
        paired_deltas=(),
        reward_diagnostics=(),
    )
    run_dir = tmp_path / "run_example"
    run_dir.mkdir()
    config = replace(
        AppConfig(),
        ppo=replace(AppConfig().ppo, experiment_tag="v5_test"),
    )
    registry_path = tmp_path / "ppo_history.csv"
    registry = register_ppo_evaluation(
        result,
        run_dir=run_dir,
        config=config,
        acceptance={"passes_all": True},
        registry_path=registry_path,
    )
    assert len(registry) == 3
    assert {row["experiment_tag"] for row in registry} == {"v5_test"}
    leaderboard = build_ppo_leaderboard(registry)
    assert leaderboard[0]["experiment_tag"] == "v5_test"
    assert leaderboard[0]["passes_all"] is True

    baseline_summary = []
    for scenario in scenarios:
        baseline_summary.append(
            {
                "scenario": scenario,
                "strategy": "cash",
                "net_pnl_mean": 10_000.0,
                "total_return_mean": 0.01,
            }
        )
    figure = plot_versioned_pnl_comparison(baseline_summary, registry)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].patches) == 6
