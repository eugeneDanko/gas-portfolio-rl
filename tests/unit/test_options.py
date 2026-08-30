from dataclasses import replace

import numpy as np
import pytest

from gas_portfolio_rl import AppConfig, GasPortfolioEnv, OptionConfig
from gas_portfolio_rl.portfolio.options import european_put_quote
from gas_portfolio_rl.training import PPOPortfolioActionWrapper, map_policy_action


def _option_config() -> AppConfig:
    base = AppConfig()
    return replace(
        base,
        option=OptionConfig(enabled=True),
        market=replace(base.market, episode_steps=80),
        ppo=replace(base.ppo, version="v4", decision_interval=5),
    )


def test_put_price_increases_with_implied_volatility_and_maturity() -> None:
    common = dict(
        spot=100.0,
        strike=95.0,
        risk_free_rate=0.02,
        dividend_yield=0.03,
    )
    low_vol = european_put_quote(
        **common, time_to_expiry=0.25, implied_volatility=0.20
    )
    high_vol = european_put_quote(
        **common, time_to_expiry=0.25, implied_volatility=0.40
    )
    long_maturity = european_put_quote(
        **common, time_to_expiry=0.50, implied_volatility=0.20
    )
    assert high_vol.price > low_vol.price
    assert long_maturity.price > low_vol.price
    assert -1.0 <= high_vol.delta <= 0.0
    assert high_vol.theta < 0.0


def test_option_environment_has_fixed_action_and_observation_shapes() -> None:
    env = GasPortfolioEnv(_option_config())
    observation, info = env.reset(seed=7)
    assert env.action_space.shape == (5,)
    assert observation.shape == (47,)
    assert observation.shape == (len(env.observation_names),)
    assert [row["tenor_days"] for row in info["option_positions"]] == [10, 21, 63]


def test_option_purchase_is_marked_in_nav_without_double_counting_premium() -> None:
    env = GasPortfolioEnv(_option_config())
    env.reset(seed=11)
    _, reward, terminated, truncated, info = env.step(
        np.asarray([0.50, 0.0, 0.40, 0.30, 0.30], dtype=np.float32)
    )
    assert not terminated and not truncated
    assert np.isfinite(reward)
    assert info["option_value"] > 0.0
    assert info["option_premium_paid"] > 0.0
    assert sum(info["option_coverages"]) <= 1.0 + 1e-8
    assert info["cash"] >= info["margin_required"]
    assert info["reward_components"]["pnl_decomposition_error"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_expired_short_put_is_cash_settled_and_reissued() -> None:
    env = GasPortfolioEnv(_option_config())
    _, info = env.reset(seed=19)
    first_expiry = info["option_positions"][0]["expiry_day"]
    action = np.asarray([0.50, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(12):
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
    short = info["option_positions"][0]
    assert short["expiry_day"] > first_expiry
    assert info["cumulative_option_payoff"] >= 0.0


def test_v4_policy_action_splits_total_coverage_with_softmax() -> None:
    mapped = map_policy_action(
        np.asarray([0.0, 0.0, 1.0, 1.0, -1.0, -1.0]),
        "v4",
    )
    assert mapped.shape == (5,)
    assert mapped[0] == pytest.approx(0.50)
    assert mapped[1] == pytest.approx(0.0)
    assert np.sum(mapped[2:]) == pytest.approx(1.0)
    assert mapped[2] > mapped[3]
    assert mapped[2] > mapped[4]


def test_v4_wrapper_holds_option_positions_between_decisions() -> None:
    config = _option_config()
    raw_env = GasPortfolioEnv(config)
    env = PPOPortfolioActionWrapper(raw_env, config.ppo)
    env.reset(seed=23)
    _, _, terminated, truncated, info = env.step(
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    )
    assert not terminated and not truncated
    assert info["decision_substeps"] == 5
    assert len(info["decision_option_coverages"]) == 3
    assert sum(info["decision_option_coverages"]) == pytest.approx(0.25)
    material_option_trades = [
        row for row in raw_env.trajectory[1:] if float(row["option_trade_value"]) > 1.0
    ]
    assert len(material_option_trades) == 1
