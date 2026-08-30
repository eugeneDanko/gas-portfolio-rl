from __future__ import annotations

from typing import Any

from gymnasium.utils.env_checker import check_env
import numpy as np

from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv


def run_preflight_checks(config: AppConfig, *, seed: int = 4242, steps: int = 64) -> dict[str, Any]:
    """Fail-fast checks required before spending compute on PPO."""
    check_env(GasPortfolioEnv(config), skip_render_check=True)
    first = GasPortfolioEnv(config)
    second = GasPortfolioEnv(config)
    observation_a, info_a = first.reset(seed=seed)
    observation_b, info_b = second.reset(seed=seed)
    deterministic = bool(np.array_equal(observation_a, observation_b) and info_a == info_b)
    finite = bool(np.all(np.isfinite(observation_a)))
    action = np.asarray([0.5, -0.25], dtype=np.float32)
    completed_steps = 0
    for _ in range(steps):
        transition_a = first.step(action)
        transition_b = second.step(action)
        observation_a, reward_a, terminated_a, truncated_a, info_a = transition_a
        observation_b, reward_b, terminated_b, truncated_b, info_b = transition_b
        deterministic &= bool(
            np.array_equal(observation_a, observation_b)
            and reward_a == reward_b
            and terminated_a == terminated_b
            and truncated_a == truncated_b
            and info_a == info_b
        )
        finite &= bool(np.all(np.isfinite(observation_a)) and np.isfinite(reward_a))
        completed_steps += 1
        if terminated_a or truncated_a:
            break
    result = {
        "gymnasium_check": True,
        "seed_determinism": deterministic,
        "finite_observations_and_rewards": finite,
        "checked_steps": completed_steps,
        "observation_shape": list(first.observation_space.shape),
        "action_shape": list(first.action_space.shape),
    }
    if not all((result["gymnasium_check"], deterministic, finite)):
        raise RuntimeError(f"Preflight checks failed: {result}")
    return result
