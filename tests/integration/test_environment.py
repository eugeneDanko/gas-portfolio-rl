from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gas_portfolio_rl import AppConfig, GasPortfolioEnv
from gas_portfolio_rl.portfolio.accounting import AccountingResult
from gas_portfolio_rl.environments.observations import OBSERVATION_NAMES


def test_environment_passes_gymnasium_checker():
    check_env(GasPortfolioEnv(), skip_render_check=True)


def test_observation_contains_calendar_delivery_and_curve_features():
    env = GasPortfolioEnv()
    observation, info = env.reset(seed=41)
    expected = {
        "sin_day_of_year",
        "cos_day_of_year",
        "sin_delivery_day",
        "cos_delivery_day",
        "futures_basis",
        "curve_slope_1_3",
        "curve_slope_3_6",
        "curve_curvature",
        "winter_summer_spread",
        "time_to_dividend",
    }
    assert expected.issubset(OBSERVATION_NAMES)
    assert observation.shape == (len(OBSERVATION_NAMES),)
    assert info["calendar_date"] is not None
    assert info["delivery_date"] is not None
    assert len(info["forward_curve"]) == 12
    assert len(info["forward_delivery_dates"]) == 12


def test_environment_is_reproducible_for_fixed_actions():
    first = GasPortfolioEnv()
    second = GasPortfolioEnv()
    observation_a, _ = first.reset(seed=123)
    observation_b, _ = second.reset(seed=123)
    np.testing.assert_array_equal(observation_a, observation_b)
    actions = [np.asarray([0.6, -0.4], dtype=np.float32)] * 10
    for action in actions:
        output_a = first.step(action)
        output_b = second.step(action)
        np.testing.assert_array_equal(output_a[0], output_b[0])
        assert output_a[1:4] == output_b[1:4]
        assert output_a[4]["portfolio_value"] == output_b[4]["portfolio_value"]


def test_action_is_projected_to_margin_feasible_region():
    env = GasPortfolioEnv()
    env.reset(seed=5)
    _, _, _, _, info = env.step(np.asarray([1.0, 1.0], dtype=np.float32))
    assert info["action_projected"]
    assert info["cash"] + 1e-8 >= info["margin_required"]


def test_time_limit_uses_truncated_not_terminated():
    base = AppConfig()
    config = replace(base, market=replace(base.market, episode_steps=2))
    env = GasPortfolioEnv(config)
    env.reset(seed=11)
    _, _, terminated, truncated, _ = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    assert not terminated and truncated


def test_margin_call_terminates_episode(monkeypatch):
    env = GasPortfolioEnv()
    env.reset(seed=19)
    original_settle = env.portfolio.settle

    def settle_with_margin_call(transition):
        result = original_settle(transition)
        return AccountingResult(
            result.stock_pnl,
            result.futures_pnl,
            result.cash_interest,
            result.forced_liquidation_cost,
            True,
        )

    monkeypatch.setattr(env.portfolio, "settle", settle_with_margin_call)
    _, reward, terminated, truncated, info = env.step(
        np.asarray([0.5, -0.5], dtype=np.float32)
    )
    assert terminated and not truncated
    assert np.isfinite(reward)
    assert info["reward_components"]["margin_call_penalty"] == pytest.approx(
        env.config.reward.margin_call_penalty
    )
    assert info["termination_reason"] == "margin_call"
