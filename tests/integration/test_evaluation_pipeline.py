from dataclasses import replace

import pytest

from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.baselines import run_strategy
from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv
from gas_portfolio_rl.evaluation import (
    build_reward_gate,
    evaluate_strategies,
    make_default_strategies,
    make_ppo_feasible_grid_strategies,
)
from gas_portfolio_rl.evaluation.reporting import save_evaluation


def test_paired_evaluation_runs_every_strategy_on_every_seed(tmp_path) -> None:
    base = AppConfig()
    config = replace(
        base,
        market=replace(base.market, episode_steps=25),
        evaluation=replace(base.evaluation, bootstrap_samples=10),
    )
    strategies = make_default_strategies(config)
    result = evaluate_strategies({"base": config}, strategies, seeds=[11, 12])
    assert len(result.records) == len(strategies) * 2
    assert {row["seed"] for row in result.records} == {11, 12}
    assert len(result.summary) == len(strategies)
    assert len(result.paired_deltas) == len(strategies) - 1
    assert len(result.reward_diagnostics) == 1
    assert "ppo_feasible_reward_utility_correlation" in result.reward_diagnostics[0]
    stress_only = evaluate_strategies(
        {"crisis_stress": config}, strategies, seeds=[11]
    )
    save_evaluation(stress_only, tmp_path)
    assert (tmp_path / "baseline_distributions.png").exists()


def test_reward_grid_covers_v3_action_set_and_builds_gate() -> None:
    base = AppConfig()
    config = replace(
        base,
        market=replace(base.market, episode_steps=20),
        evaluation=replace(base.evaluation, bootstrap_samples=5),
    )
    scenarios = {name: config for name in ("base", "calm", "crisis_stress")}
    baseline = evaluate_strategies(
        scenarios, make_default_strategies(config), seeds=[11, 12]
    )
    grid_strategies = make_ppo_feasible_grid_strategies(config)
    grid = evaluate_strategies(scenarios, grid_strategies, seeds=[11, 12])
    gate, ranking = build_reward_gate(baseline, grid, config)

    assert len(grid_strategies) == 15
    assert len(ranking) == 15
    assert set(gate["checks"]) == {
        "baseline_ppo_feasible_alignment",
        "grid_ppo_feasible_alignment",
        "robust_optimum_has_no_margin_calls",
        "robust_optimum_has_no_bankruptcies",
    }
    assert set(gate["diagnostics"]) == {
        "robust_optimum_has_interior_stock",
        "robust_optimum_uses_futures",
        "crisis_hedge_improves_same_stock_reward",
    }


def test_reward_grid_covers_v4_option_axes_and_builds_gate() -> None:
    base = AppConfig()
    config = replace(
        base,
        market=replace(base.market, episode_steps=10),
        option=replace(base.option, enabled=True),
        ppo=replace(base.ppo, version="v4"),
        evaluation=replace(base.evaluation, bootstrap_samples=3),
    )
    strategies = make_ppo_feasible_grid_strategies(config)
    assert len(strategies) == 54
    assert any("oshort100" in name for name in strategies)
    assert any("omedium100" in name for name in strategies)
    assert any("olong100" in name for name in strategies)
    assert any("oladder050" in name for name in strategies)
    assert any("oladder100" in name for name in strategies)

    scenarios = {name: config for name in ("base", "calm", "crisis_stress")}
    baseline = evaluate_strategies(
        scenarios, make_default_strategies(config), seeds=[11]
    )
    grid = evaluate_strategies(scenarios, strategies, seeds=[11])
    gate, ranking = build_reward_gate(baseline, grid, config)

    assert len(ranking) == 54
    assert "total_option_coverage" in ranking[0]
    assert "robust_optimum_uses_options" in gate["diagnostics"]


def test_v4_grid_restores_long_put_after_expiry() -> None:
    base = AppConfig()
    config = replace(
        base,
        market=replace(base.market, episode_steps=80),
        option=replace(base.option, enabled=True),
        ppo=replace(base.ppo, version="v4", decision_interval=5),
    )
    strategy = make_ppo_feasible_grid_strategies(config)[
        "grid_s025_fp000_olong100"
    ]()

    run = run_strategy(GasPortfolioEnv(config), strategy, seed=42)
    long_coverages = [
        float(row["option_coverages"][2]) for row in run.trajectory
    ]

    assert long_coverages[63] == pytest.approx(0.0)
    assert max(long_coverages[64:70]) > 0.90
    assert sum(long_coverages) / len(long_coverages) > 0.80
