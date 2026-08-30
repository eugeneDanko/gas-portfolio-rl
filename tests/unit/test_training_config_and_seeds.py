from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gas_portfolio_rl import AppConfig, PPOConfig
from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv
from gas_portfolio_rl.training import EpisodeSeedWrapper


def test_ppo_config_loads_from_yaml() -> None:
    path = Path(__file__).parents[2] / "configs" / "models" / "ppo.yaml"
    config = AppConfig.from_yaml(path)
    assert config.ppo == replace(PPOConfig(), version="v4")
    assert config.ppo.policy_layers == (128, 128)


def test_frozen_v1_config_remains_loadable() -> None:
    path = Path(__file__).parents[2] / "configs" / "models" / "ppo_v1.yaml"
    config = AppConfig.from_yaml(path)
    assert config.ppo.version == "v1"
    assert config.ppo.decision_interval == 1
    assert config.ppo.gamma == 0.995


def test_resolved_run_config_round_trips_from_metadata_mapping() -> None:
    expected = AppConfig()
    restored = AppConfig.from_mapping(asdict(expected))
    assert restored == expected


def test_ppo_config_requires_batch_to_divide_rollout() -> None:
    with pytest.raises(ValueError, match="divide"):
        PPOConfig(n_envs=2, n_steps=100, batch_size=128)


def test_episode_seed_wrapper_is_bounded_and_reproducible() -> None:
    first = EpisodeSeedWrapper(
        GasPortfolioEnv(), seed_start=1_000, seed_end=1_005, selector_seed=77
    )
    second = EpisodeSeedWrapper(
        GasPortfolioEnv(), seed_start=1_000, seed_end=1_005, selector_seed=77
    )
    first_seeds = [first.reset()[1]["episode_seed"] for _ in range(6)]
    second_seeds = [second.reset()[1]["episode_seed"] for _ in range(6)]
    assert first_seeds == second_seeds
    assert all(1_000 <= seed <= 1_005 for seed in first_seeds)
    assert not any(10_000 <= seed <= 29_999 for seed in first_seeds)
