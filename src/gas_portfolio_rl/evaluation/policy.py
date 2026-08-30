from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from gas_portfolio_rl.config.models import PPOConfig
from gas_portfolio_rl.training.action_wrapper import (
    apply_action_deadband,
    enforce_hedge_constraints,
    map_policy_action,
    smooth_option_targets,
    wrap_ppo_env,
)


class NormalizedModelStrategy:
    """Run an SB3 model against a raw environment using frozen VecNormalize stats."""

    def __init__(
        self,
        model: Any,
        normalizer: Any,
        *,
        ppo_config: PPOConfig | None = None,
        name: str = "ppo",
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.name = name
        self.deterministic = deterministic
        self.ppo_config = ppo_config
        self._days_until_decision = 0

    def reset(self) -> None:
        self._days_until_decision = 0

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        if self.ppo_config is not None and self.ppo_config.version in {"v2", "v3", "v4"}:
            if self._days_until_decision > 0:
                self._days_until_decision -= 1
                held = [float(info["stock_weight"]), float(info["futures_exposure"])]
                if self.ppo_config.version == "v4":
                    held.extend(float(value) for value in info.get("option_coverages", (0, 0, 0)))
                return np.asarray(held, dtype=np.float32)
        normalized = self.normalizer.normalize_obs(observation[None, :])[0]
        action, _ = self.model.predict(normalized, deterministic=self.deterministic)
        if self.ppo_config is None or self.ppo_config.version == "v1":
            return np.asarray(action, dtype=np.float32)
        candidate = map_policy_action(
            action,
            self.ppo_config.version,
            minimum_stock_weight=self.ppo_config.minimum_stock_weight,
            maximum_stock_weight=self.ppo_config.maximum_stock_weight,
            maximum_futures_exposure=self.ppo_config.maximum_futures_exposure,
            minimum_option_coverage=self.ppo_config.minimum_option_coverage,
            maximum_option_coverage=1.0,
        )
        candidate = smooth_option_targets(
            candidate,
            info,
            smoothing=self.ppo_config.option_target_smoothing,
        )
        selected = apply_action_deadband(
            candidate,
            info,
            stock_deadband=self.ppo_config.stock_action_deadband,
            futures_deadband=self.ppo_config.futures_action_deadband,
            option_deadband=self.ppo_config.option_coverage_deadband,
        )
        if self.ppo_config.version in {"v3", "v4"}:
            selected = enforce_hedge_constraints(selected, self.ppo_config)
        self._days_until_decision = self.ppo_config.decision_interval - 1
        return selected


def load_policy_bundle(
    model_path: str,
    normalization_path: str,
    config: Any,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv

    dummy = DummyVecEnv(
        [lambda: wrap_ppo_env(GasPortfolioEnv(config), config.ppo)]
    )
    normalizer = VecNormalize.load(normalization_path, dummy)
    normalizer.training = False
    normalizer.norm_reward = False
    model = PPO.load(model_path, env=normalizer, device="auto")
    return model, normalizer
