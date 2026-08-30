from pathlib import Path

import numpy as np
import pytest

from gas_portfolio_rl import (
    AppConfig,
    RelativeMomentumConfig,
    RollingHedgeConfig,
    SignedRollingHedgeConfig,
)
from gas_portfolio_rl.baselines import (
    ConstantAllocationBaseline,
    RelativeMomentumBaseline,
    RollingHedgeBaseline,
    SignedRollingHedgeBaseline,
)


def test_constant_allocation_restores_option_target_at_rebalance():
    strategy = ConstantAllocationBaseline(
        stock_weight=0.25,
        futures_exposure=0.0,
        name="long_put_grid",
        option_coverages=(0.0, 0.0, 1.0),
        rebalance_interval=5,
    )
    info = {
        "step": 65,
        "option_enabled": True,
        "stock_weight": 0.24,
        "futures_exposure": 0.0,
        "option_coverages": [0.0, 0.0, 0.0],
    }

    action = strategy.act(np.empty(0), info)

    assert action.tolist() == pytest.approx([0.24, 0.0, 0.0, 0.0, 1.0])


def test_constant_allocation_holds_options_between_rebalances():
    strategy = ConstantAllocationBaseline(
        stock_weight=0.25,
        futures_exposure=0.0,
        name="long_put_grid",
        option_coverages=(0.0, 0.0, 1.0),
        rebalance_interval=5,
    )
    info = {
        "step": 63,
        "option_enabled": True,
        "stock_weight": 0.24,
        "futures_exposure": 0.0,
        "option_coverages": [0.0, 0.0, 0.0],
    }

    action = strategy.act(np.empty(0), info)

    assert action.tolist() == pytest.approx([0.24, 0.0, 0.0, 0.0, 0.0])


def _info(
    step: int,
    stock: float,
    future: float,
    dividend_per_share: float = 0.0,
) -> dict[str, float | int]:
    return {
        "step": step,
        "stock_price": stock,
        "futures_price": future,
        "stock_dividend_per_share": dividend_per_share,
    }


def _feed(strategy, stock_prices, futures_prices, dividends=None):
    dividends = dividends or [0.0] * len(stock_prices)
    action = None
    for step, (stock, future, dividend) in enumerate(
        zip(stock_prices, futures_prices, dividends, strict=True)
    ):
        action = strategy.act(np.empty(0), _info(step, stock, future, dividend))
    return action


def test_relative_momentum_selects_stronger_stock():
    strategy = RelativeMomentumBaseline(
        RelativeMomentumConfig(lookback=3, score_threshold=0.0, rebalance_interval=1)
    )
    action = _feed(strategy, [100, 101, 102, 103], [10, 10, 10, 10])
    np.testing.assert_allclose(action, [0.5, 0.0])


def test_relative_momentum_selects_stronger_future():
    strategy = RelativeMomentumBaseline(
        RelativeMomentumConfig(lookback=3, score_threshold=0.0, rebalance_interval=1)
    )
    action = _feed(strategy, [100, 100, 100, 100], [10, 10.1, 10.2, 10.3])
    np.testing.assert_allclose(action, [0.0, 0.5])


def test_relative_momentum_uses_dividend_adjusted_total_return():
    strategy = RelativeMomentumBaseline(
        RelativeMomentumConfig(lookback=3, score_threshold=0.0, rebalance_interval=1)
    )
    action = _feed(
        strategy,
        [100.0, 99.75, 99.75, 99.75],
        [10.0, 10.0, 10.0, 10.0],
        [0.0, 0.25, 0.0, 0.0],
    )

    assert strategy.last_scores[0] == pytest.approx(0.0)
    np.testing.assert_allclose(action, [0.25, 0.0])


def test_rolling_hedge_estimates_minimum_variance_exposure():
    config = RollingHedgeConfig(lookback=3, rebalance_interval=1)
    strategy = RollingHedgeBaseline(config)
    futures_returns = np.asarray([0.01, -0.005, 0.015])
    stock_returns = 0.5 * futures_returns
    futures_prices = 10.0 * np.exp(np.r_[0.0, np.cumsum(futures_returns)])
    stock_prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(stock_returns)])

    action = _feed(strategy, stock_prices, futures_prices)

    assert strategy.last_hedge_beta == pytest.approx(0.5)
    np.testing.assert_allclose(action, [0.5, -0.25], atol=1e-6)


@pytest.mark.parametrize(
    ("stock_beta", "expected_exposure"),
    [(0.5, -0.25), (-0.5, 0.25)],
)
def test_signed_rolling_hedge_uses_both_exposure_directions(
    stock_beta, expected_exposure
):
    config = SignedRollingHedgeConfig(
        lookback=3,
        rebalance_interval=1,
        beta_smoothing=1.0,
        beta_shrinkage=1.0,
        minimum_abs_correlation=0.0,
        exposure_change_threshold=0.0,
    )
    strategy = SignedRollingHedgeBaseline(config)
    futures_returns = np.asarray([0.01, -0.005, 0.015])
    stock_returns = stock_beta * futures_returns
    futures_prices = 10.0 * np.exp(np.r_[0.0, np.cumsum(futures_returns)])
    stock_prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(stock_returns)])

    action = _feed(strategy, stock_prices, futures_prices)

    assert strategy.last_raw_beta == pytest.approx(stock_beta)
    np.testing.assert_allclose(action, [0.5, expected_exposure], atol=1e-6)


def test_signed_rolling_hedge_ignores_weak_correlation():
    config = SignedRollingHedgeConfig(
        lookback=4,
        rebalance_interval=1,
        beta_smoothing=1.0,
        beta_shrinkage=1.0,
        minimum_abs_correlation=0.95,
        exposure_change_threshold=0.0,
    )
    strategy = SignedRollingHedgeBaseline(config)
    futures_returns = np.asarray([0.01, -0.01, 0.01, -0.01])
    stock_returns = np.asarray([0.01, 0.01, -0.01, -0.01])
    action = _feed(
        strategy,
        100.0 * np.exp(np.r_[0.0, np.cumsum(stock_returns)]),
        10.0 * np.exp(np.r_[0.0, np.cumsum(futures_returns)]),
    )
    np.testing.assert_allclose(action, [0.5, 0.0], atol=1e-6)


def test_signed_rolling_hedge_deadband_holds_actual_positions():
    strategy = SignedRollingHedgeBaseline(
        SignedRollingHedgeConfig(
            lookback=3,
            rebalance_interval=1,
            beta_smoothing=1.0,
            beta_shrinkage=1.0,
            minimum_abs_correlation=0.0,
            exposure_change_threshold=0.10,
        )
    )
    futures_returns = np.asarray([0.01, -0.005, 0.015])
    stock_returns = 0.5 * futures_returns
    futures_prices = 10.0 * np.exp(np.r_[0.0, np.cumsum(futures_returns)])
    stock_prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(stock_returns)])
    for step in range(3):
        strategy.act(np.empty(0), _info(step, stock_prices[step], futures_prices[step]))
    final_info = _info(3, stock_prices[3], futures_prices[3])
    final_info.update(stock_weight=0.47, futures_exposure=-0.21)
    action = strategy.act(np.empty(0), final_info)
    np.testing.assert_allclose(action, [0.47, -0.21])


def test_baseline_config_loads_from_yaml():
    path = Path(__file__).parents[2] / "configs" / "baselines" / "default.yaml"
    config = AppConfig.from_yaml(path)
    assert config.baselines.relative_momentum == RelativeMomentumConfig()
    assert config.baselines.rolling_hedge == RollingHedgeConfig()
    assert config.baselines.signed_rolling_hedge == SignedRollingHedgeConfig()
