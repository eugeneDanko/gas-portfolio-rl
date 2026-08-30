"""Synthetic gas portfolio RL environment."""

from gas_portfolio_rl.config.models import (
    AppConfig,
    BaselinesConfig,
    CalibrationConfig,
    EnvConfig,
    EvaluationConfig,
    MarketConfig,
    OptionConfig,
    PortfolioConfig,
    PPOConfig,
    RelativeMomentumConfig,
    RewardConfig,
    RollingHedgeConfig,
    SignedRollingHedgeConfig,
)

__all__ = [
    "AppConfig",
    "BaselinesConfig",
    "CalibrationConfig",
    "EnvConfig",
    "EvaluationConfig",
    "GasPortfolioEnv",
    "MarketConfig",
    "OptionConfig",
    "PortfolioConfig",
    "PPOConfig",
    "RelativeMomentumConfig",
    "RewardConfig",
    "RollingHedgeConfig",
    "SignedRollingHedgeConfig",
]


def __getattr__(name: str):
    if name == "GasPortfolioEnv":
        from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv

        return GasPortfolioEnv
    raise AttributeError(name)
