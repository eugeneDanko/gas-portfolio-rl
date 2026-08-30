from dataclasses import replace

import numpy as np
import pytest

from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv
from gas_portfolio_rl.training import (
    PPOPortfolioActionWrapper,
    apply_action_deadband,
    map_policy_action,
    smooth_option_targets,
)


def test_v2_affine_stock_action_has_no_negative_dead_zone() -> None:
    np.testing.assert_allclose(map_policy_action(np.array([-1.0, -0.4])), [0.0, -0.4])
    np.testing.assert_allclose(map_policy_action(np.array([0.0, 0.0])), [0.5, 0.0])
    np.testing.assert_allclose(map_policy_action(np.array([1.0, 0.4])), [1.0, 0.4])


def test_v3_action_requires_stock_and_caps_futures_by_insured_exposure() -> None:
    np.testing.assert_allclose(
        map_policy_action(np.array([-1.0, 1.0]), "v3"), [0.25, 0.25]
    )
    np.testing.assert_allclose(
        map_policy_action(np.array([0.0, -1.0]), "v3"), [0.50, -0.50]
    )
    np.testing.assert_allclose(
        map_policy_action(np.array([1.0, 1.0]), "v3"), [0.75, 0.50]
    )


def test_action_deadband_keeps_current_weights() -> None:
    selected = apply_action_deadband(
        np.array([0.51, -0.12]),
        {"stock_weight": 0.50, "futures_exposure": -0.10},
        stock_deadband=0.03,
        futures_deadband=0.03,
    )
    np.testing.assert_allclose(selected, [0.50, -0.10])


def test_v4_option_target_respects_floor_and_is_smoothed() -> None:
    mapped = map_policy_action(
        np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
        "v4",
        minimum_option_coverage=0.25,
    )
    assert np.sum(mapped[2:]) == pytest.approx(0.25)
    smoothed = smooth_option_targets(
        np.asarray([0.5, 0.0, 0.6, 0.2, 0.2]),
        {"option_coverages": [0.0, 0.5, 0.5]},
        smoothing=0.25,
    )
    np.testing.assert_allclose(smoothed[2:], [0.15, 0.425, 0.425])


def test_v2_wrapper_advances_one_decision_interval_and_holds() -> None:
    base = AppConfig()
    config = replace(
        base,
        market=replace(base.market, episode_steps=20),
        ppo=replace(base.ppo, decision_interval=5),
    )
    raw_env = GasPortfolioEnv(config)
    env = PPOPortfolioActionWrapper(raw_env, config.ppo)
    env.reset(seed=42)

    _, _, terminated, truncated, info = env.step(np.array([0.0, -0.2]))

    assert not terminated and not truncated
    assert info["step"] == 5
    assert info["decision_substeps"] == 5
    assert info["decision_stock_target"] == 0.5
    # Only the first substep should create a material stock trade.
    material_stock_trades = [
        row for row in raw_env.trajectory[1:] if abs(float(row["stock_trade_value"])) > 1.0
    ]
    assert len(material_stock_trades) == 1
