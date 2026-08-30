from pathlib import Path

import pytest

from gas_portfolio_rl import AppConfig, RewardConfig
from gas_portfolio_rl.environments.rewards import calculate_reward


def _reward(**overrides):
    cash_interest = 1_000.0 * 0.02 / 252
    values = dict(
        value_before_action=1_000.0,
        value_after_step=1_000.0 + cash_interest,
        initial_value=1_000.0,
        stock_pnl=0.0,
        futures_pnl=0.0,
        cash_interest=cash_interest,
        stock_dividend=0.0,
        stock_transaction_cost=0.0,
        futures_transaction_cost=0.0,
        forced_liquidation_cost=0.0,
        turnover=0.0,
        previous_drawdown=0.0,
        drawdown=0.0,
        margin_utilization=0.0,
        annual_cash_rate=0.02,
        margin_call=False,
        bankrupt=False,
        trading_days=252,
        current_step=100,
        episode_steps=504,
        config=RewardConfig(),
    )
    values.update(overrides)
    return calculate_reward(**values)


def test_cash_benchmark_has_zero_reward_and_exact_pnl_decomposition():
    reward, components = _reward()
    assert reward == pytest.approx(0.0, abs=1e-12)
    assert components["excess_return"] == pytest.approx(0.0, abs=1e-12)
    assert components["pnl_decomposition_error"] == pytest.approx(0.0, abs=1e-12)


def test_losses_have_larger_utility_slope_than_gains():
    cash_return = 0.02 / 252
    gain, gain_components = _reward(value_after_step=1_000.0 * (1 + cash_return + 0.01))
    loss, loss_components = _reward(value_after_step=1_000.0 * (1 + cash_return - 0.01))
    assert gain_components["profit_reward"] == pytest.approx(0.01)
    assert loss_components["profit_reward"] == pytest.approx(-0.0125)
    assert loss < -gain


def test_futures_receive_credit_only_when_they_reduce_downside():
    cash_return = 0.02 / 252
    _, hedged = _reward(
        value_after_step=1_000.0 * (1 + cash_return - 0.01),
        futures_pnl=20.0,
    )
    _, harmful = _reward(
        value_after_step=1_000.0 * (1 + cash_return - 0.01),
        futures_pnl=-20.0,
    )
    assert hedged["hedge_effectiveness"] == pytest.approx(0.02)
    assert hedged["hedge_reward"] > 0
    assert harmful["hedge_reward"] < 0


def test_options_share_the_same_counterfactual_hedge_credit():
    cash_return = 0.02 / 252
    _, hedged = _reward(
        value_after_step=1_000.0 * (1 + cash_return - 0.01),
        option_pnl=20.0,
    )
    _, harmful = _reward(
        value_after_step=1_000.0 * (1 + cash_return - 0.01),
        option_pnl=-20.0,
    )
    assert hedged["option_return_contribution"] == pytest.approx(0.02)
    assert hedged["hedge_reward"] > 0
    assert harmful["hedge_reward"] < 0


def test_option_carry_is_charged_during_a_positive_market_day():
    cash_return = 0.02 / 252
    _, components = _reward(
        value_after_step=1_000.0 * (1 + cash_return + 0.01),
        option_pnl=-20.0,
    )
    assert components["excess_return"] == pytest.approx(0.01)
    assert components["unhedged_excess_return"] == pytest.approx(0.03)
    assert components["hedge_effectiveness"] == pytest.approx(-0.02)
    assert components["hedge_reward"] < 0.0


def test_counterfactual_hedge_uses_loss_averse_utility_symmetrically():
    cash_return = 0.02 / 252
    _, protected = _reward(
        value_after_step=1_000.0 * (1 + cash_return - 0.01),
        option_pnl=10.0,
    )
    _, costly = _reward(
        value_after_step=1_000.0 * (1 + cash_return + 0.01),
        option_pnl=-10.0,
    )
    assert protected["hedge_effectiveness"] == pytest.approx(0.0125)
    assert costly["hedge_effectiveness"] == pytest.approx(-0.01)


def test_incremental_drawdown_turnover_and_margin_are_penalized():
    reward, components = _reward(
        previous_drawdown=-0.10,
        drawdown=-0.13,
        turnover=0.50,
        margin_utilization=0.18,
    )
    assert components["drawdown_penalty"] == pytest.approx(0.003)
    assert components["turnover_penalty"] == pytest.approx(0.001)
    assert components["margin_utilization_penalty"] == pytest.approx(2.0 * 0.10**2)
    assert reward == pytest.approx(-0.024)


def test_terminal_shortfall_and_drawdown_budget_apply_only_at_horizon():
    _, before_terminal = _reward(
        value_after_step=800.0, drawdown=-0.30, current_step=503
    )
    _, terminal = _reward(
        value_after_step=800.0, drawdown=-0.30, current_step=504
    )
    assert before_terminal["terminal_shortfall_penalty"] == 0.0
    assert before_terminal["terminal_drawdown_penalty"] == 0.0
    assert terminal["terminal_shortfall_penalty"] > 0.0
    assert terminal["terminal_drawdown_penalty"] == pytest.approx(0.30)


def test_terminal_penalty_remembers_historical_drawdown_after_recovery():
    _, components = _reward(
        drawdown=0.0,
        previous_max_drawdown=0.30,
        max_drawdown=0.30,
        current_step=504,
    )
    assert components["drawdown_budget_penalty"] == 0.0
    assert components["terminal_drawdown_penalty"] == pytest.approx(0.30)


def test_drawdown_budget_penalty_is_dense_but_recoverable():
    _, breached = _reward(
        previous_drawdown=-0.15,
        drawdown=-0.25,
        previous_max_drawdown=0.15,
        max_drawdown=0.25,
    )
    _, recovered = _reward(
        previous_drawdown=-0.25,
        drawdown=0.0,
        previous_max_drawdown=0.25,
        max_drawdown=0.25,
    )
    assert breached["drawdown_budget_penalty"] == pytest.approx(0.5 * 0.10**2)
    assert recovered["drawdown_budget_penalty"] == 0.0
    assert recovered["max_drawdown"] == pytest.approx(0.25)


def test_earlier_margin_call_is_penalized_more_than_later_one():
    early, early_components = _reward(margin_call=True, current_step=100)
    late, late_components = _reward(margin_call=True, current_step=500)
    assert early < late
    assert early_components["early_termination_penalty"] > late_components[
        "early_termination_penalty"
    ]


def test_bankruptcy_is_finite_and_more_severe_than_margin_call():
    margin, _ = _reward(value_after_step=100.0, margin_call=True)
    bankruptcy, components = _reward(value_after_step=-100.0, bankrupt=True)
    assert bankruptcy < margin
    assert components["bankruptcy_penalty"] == pytest.approx(10.2)


def test_excess_return_is_scale_invariant():
    first, first_components = _reward(
        value_before_action=1_000.0, value_after_step=1_010.0
    )
    second, second_components = _reward(
        value_before_action=1_000_000.0,
        value_after_step=1_010_000.0,
        initial_value=1_000_000.0,
        cash_interest=1_000_000.0 * 0.02 / 252,
    )
    assert first_components["excess_return"] == pytest.approx(
        second_components["excess_return"]
    )
    assert first == pytest.approx(second)


def test_reward_config_is_loaded_from_separate_yaml():
    path = Path(__file__).parents[2] / "configs" / "reward" / "default.yaml"
    config = AppConfig.from_yaml(path)
    assert config.reward == RewardConfig()


def test_reward_config_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="non-negative"):
        RewardConfig(downside_penalty=-1.0)
    with pytest.raises(ValueError, match="at least 1"):
        RewardConfig(loss_aversion=0.5)
