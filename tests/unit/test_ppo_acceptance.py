from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.evaluation.acceptance import assess_ppo
from gas_portfolio_rl.evaluation.evaluator import EvaluationResult


def _summary(
    scenario,
    strategy,
    reward,
    drawdown,
    high_cash=0.2,
    total_return=0.10,
    stock_weight=0.30,
    turnover=5.0,
):
    return {
        "scenario": scenario,
        "strategy": strategy,
        "total_reward_mean": reward,
        "max_drawdown_mean": drawdown,
        "margin_call_rate": 0.0,
        "bankruptcy_rate": 0.0,
        "high_cash_fraction_mean": high_cash,
        "total_return_mean": total_return,
        "average_stock_weight_mean": stock_weight,
        "turnover_mean": turnover,
    }


def test_ppo_acceptance_requires_performance_safety_and_non_degeneracy() -> None:
    result = EvaluationResult(
        records=(),
        summary=(
            _summary("base", "cash", -0.16, 0.0, 1.0, 0.04, 0.0, 0.0),
            _summary("base", "minimum_stock_unhedged", -0.05, 0.05),
            _summary("base", "ppo", 0.10, 0.10),
            _summary("calm", "cash", -0.16, 0.0, 1.0, 0.04, 0.0, 0.0),
            _summary("calm", "minimum_stock_unhedged", -0.05, 0.05),
            _summary("calm", "ppo", 0.20, 0.08),
            _summary("crisis_stress", "cash", -3.0, 0.0, 1.0, 0.04, 0.0, 0.0),
            _summary("crisis_stress", "signed_rolling_hedge", -5.0, 0.40),
            _summary("crisis_stress", "minimum_stock_unhedged", -3.0, 0.20),
            _summary("crisis_stress", "ppo", -2.0, 0.30),
        ),
        paired_deltas=(),
        reward_diagnostics=(),
    )
    acceptance = assess_ppo(result, AppConfig())
    assert acceptance["passes_all"]
    assert all(acceptance["checks"].values())


def test_ppo_acceptance_rejects_constant_cash() -> None:
    result = EvaluationResult(
        records=(),
        summary=(
            _summary("base", "cash", -0.16, 0.0, 1.0, 0.04, 0.0, 0.0),
            _summary("base", "minimum_stock_unhedged", -0.05, 0.05),
            _summary("base", "ppo", 0.10, 0.01, 1.0),
            _summary("calm", "cash", -0.16, 0.0, 1.0, 0.04, 0.0, 0.0),
            _summary("calm", "minimum_stock_unhedged", -0.05, 0.05),
            _summary("calm", "ppo", 0.20, 0.08),
            _summary("crisis_stress", "cash", -3.0, 0.0, 1.0, 0.04, 0.0, 0.0),
            _summary("crisis_stress", "signed_rolling_hedge", -5.0, 0.40),
            _summary("crisis_stress", "minimum_stock_unhedged", -3.0, 0.20),
            _summary("crisis_stress", "ppo", -2.0, 0.30, 1.0),
        ),
        paired_deltas=(),
        reward_diagnostics=(),
    )
    acceptance = assess_ppo(result, AppConfig())
    assert not acceptance["passes_all"]
    assert not acceptance["checks"]["not_constant_cash"]


def test_ppo_v2_acceptance_rejects_low_stock_and_excess_turnover() -> None:
    result = EvaluationResult(
        records=(),
        summary=(
            _summary("base", "cash", -0.16, 0.0, 1.0, 0.04, 0.0, 0.0),
            _summary("base", "minimum_stock_unhedged", -0.05, 0.05),
            _summary("base", "ppo", 0.10, 0.10, stock_weight=0.01, turnover=30.0),
            _summary("calm", "cash", -0.16, 0.0, 1.0, 0.04, 0.0, 0.0),
            _summary("calm", "minimum_stock_unhedged", -0.05, 0.05),
            _summary("calm", "ppo", 0.20, 0.08, turnover=30.0),
            _summary("crisis_stress", "cash", -3.0, 0.0, 1.0, 0.04, 0.0, 0.0),
            _summary("crisis_stress", "signed_rolling_hedge", -5.0, 0.40),
            _summary("crisis_stress", "minimum_stock_unhedged", -3.0, 0.20),
            _summary("crisis_stress", "ppo", -2.0, 0.30, turnover=30.0),
        ),
        paired_deltas=(),
        reward_diagnostics=(),
    )

    acceptance = assess_ppo(result, AppConfig())

    assert not acceptance["passes_all"]
    assert not acceptance["checks"]["minimum_stock_participation"]
    assert not acceptance["checks"]["turnover_within_limit"]
