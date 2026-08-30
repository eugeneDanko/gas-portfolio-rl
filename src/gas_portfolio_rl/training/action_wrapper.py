from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from gas_portfolio_rl.config.models import PPOConfig


def map_policy_action(
    action: np.ndarray,
    version: str = "v2",
    *,
    minimum_stock_weight: float = 0.25,
    maximum_stock_weight: float = 0.75,
    maximum_futures_exposure: float = 0.50,
    minimum_option_coverage: float = 0.0,
    maximum_option_coverage: float = 1.0,
) -> np.ndarray:
    """Map a symmetric Gaussian policy action to portfolio targets."""
    raw = np.asarray(action, dtype=np.float64).reshape(-1)
    expected_size = 6 if version == "v4" else 2
    if raw.size != expected_size or not np.all(np.isfinite(raw)):
        raise ValueError(f"PPO {version} action must contain {expected_size} finite values")
    clipped = np.clip(raw, -1.0, 1.0)
    if version == "v1":
        stock_weight = np.clip(clipped[0], 0.0, 1.0)
    elif version == "v2":
        stock_weight = 0.5 * (clipped[0] + 1.0)
    elif version in {"v3", "v4"}:
        stock_weight = minimum_stock_weight + 0.5 * (clipped[0] + 1.0) * (
            maximum_stock_weight - minimum_stock_weight
        )
        futures_limit = min(stock_weight, maximum_futures_exposure)
        if version == "v4":
            if not 0 <= minimum_option_coverage <= maximum_option_coverage:
                raise ValueError("option coverage limits must satisfy 0 <= minimum <= maximum")
            total_coverage = minimum_option_coverage + 0.5 * (clipped[2] + 1.0) * (
                maximum_option_coverage - minimum_option_coverage
            )
            logits = clipped[3:6]
            weights = np.exp(logits - np.max(logits))
            weights /= np.sum(weights)
            coverages = total_coverage * weights
            return np.asarray(
                [stock_weight, clipped[1] * futures_limit, *coverages],
                dtype=np.float32,
            )
        return np.asarray(
            [stock_weight, clipped[1] * futures_limit], dtype=np.float32
        )
    else:
        raise ValueError(f"Unknown PPO action version: {version}")
    return np.asarray([stock_weight, clipped[1]], dtype=np.float32)


def enforce_hedge_constraints(
    action: np.ndarray,
    config: PPOConfig,
    *,
    maximum_option_coverage: float = 1.0,
) -> np.ndarray:
    """Project a target back to the v3 insured-portfolio action set."""
    selected = np.asarray(action, dtype=np.float64).reshape(-1)
    stock_weight = float(
        np.clip(
            selected[0], config.minimum_stock_weight, config.maximum_stock_weight
        )
    )
    futures_limit = min(stock_weight, config.maximum_futures_exposure)
    futures_exposure = float(np.clip(selected[1], -futures_limit, futures_limit))
    if selected.size == 2:
        return np.asarray([stock_weight, futures_exposure], dtype=np.float32)
    coverages = np.clip(selected[2:5], 0.0, maximum_option_coverage)
    total = float(np.sum(coverages))
    if total > maximum_option_coverage:
        coverages *= maximum_option_coverage / total
    elif 0 < total < config.minimum_option_coverage:
        coverages *= config.minimum_option_coverage / total
    elif total <= 0 and config.minimum_option_coverage > 0:
        coverages[:] = config.minimum_option_coverage / len(coverages)
    return np.asarray([stock_weight, futures_exposure, *coverages], dtype=np.float32)


def smooth_option_targets(
    candidate: np.ndarray,
    info: Mapping[str, Any],
    *,
    smoothing: float,
) -> np.ndarray:
    """Move option coverage part-way toward the new policy target."""
    selected = np.asarray(candidate, dtype=np.float32).copy()
    if selected.size != 5:
        return selected
    current = np.asarray(info.get("option_coverages", (0.0, 0.0, 0.0)), dtype=np.float32)
    if current.size != 3:
        current = np.zeros(3, dtype=np.float32)
    selected[2:5] = current + smoothing * (selected[2:5] - current)
    return selected


def apply_action_deadband(
    candidate: np.ndarray,
    info: Mapping[str, Any],
    *,
    stock_deadband: float,
    futures_deadband: float,
    option_deadband: float = 0.0,
) -> np.ndarray:
    selected = np.asarray(candidate, dtype=np.float32).copy()
    current_values = [float(info["stock_weight"]), float(info["futures_exposure"])]
    if selected.size == 5:
        current_values.extend(float(value) for value in info.get("option_coverages", (0, 0, 0)))
    current = np.asarray(current_values, dtype=np.float32)
    if abs(float(selected[0] - current[0])) < stock_deadband:
        selected[0] = current[0]
    if abs(float(selected[1] - current[1])) < futures_deadband:
        selected[1] = current[1]
    if selected.size == 5:
        for index in range(2, 5):
            if abs(float(selected[index] - current[index])) < option_deadband:
                selected[index] = current[index]
    return selected


class PPOPortfolioActionWrapper(gym.Wrapper):
    """Turn one PPO decision into a low-turnover multi-day portfolio hold."""

    def __init__(self, env: gym.Env, config: PPOConfig) -> None:
        super().__init__(env)
        self.ppo_config = config
        action_dimensions = 6 if config.version == "v4" else 2
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dimensions,),
            dtype=np.float32,
        )
        self._last_info: dict[str, Any] | None = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_info = dict(info)
        return observation, info

    def step(self, action: np.ndarray):
        if self._last_info is None:
            raise RuntimeError("Call reset() before step()")
        candidate = map_policy_action(
            action,
            self.ppo_config.version,
            minimum_stock_weight=self.ppo_config.minimum_stock_weight,
            maximum_stock_weight=self.ppo_config.maximum_stock_weight,
            maximum_futures_exposure=self.ppo_config.maximum_futures_exposure,
            minimum_option_coverage=self.ppo_config.minimum_option_coverage,
            maximum_option_coverage=self.env.unwrapped.config.option.maximum_total_coverage,
        )
        candidate = smooth_option_targets(
            candidate,
            self._last_info,
            smoothing=self.ppo_config.option_target_smoothing,
        )
        target = apply_action_deadband(
            candidate,
            self._last_info,
            stock_deadband=self.ppo_config.stock_action_deadband,
            futures_deadband=self.ppo_config.futures_action_deadband,
            option_deadband=self.ppo_config.option_coverage_deadband,
        )
        if self.ppo_config.version in {"v3", "v4"}:
            target = enforce_hedge_constraints(
                target,
                self.ppo_config,
                maximum_option_coverage=self.env.unwrapped.config.option.maximum_total_coverage,
            )
        total_reward = 0.0
        terminated = False
        truncated = False
        observation = None
        info: Mapping[str, Any] = self._last_info
        substeps = 0
        for substep in range(self.ppo_config.decision_interval):
            # Only the first day rebalances to the new target. On following days
            # the current weights are passed back, which is a true buy-and-hold.
            if substep == 0:
                held_action = target
            else:
                held_values = [float(info["stock_weight"]), float(info["futures_exposure"])]
                if self.ppo_config.version == "v4":
                    held_values.extend(
                        float(value) for value in info.get("option_coverages", (0, 0, 0))
                    )
                held_action = np.asarray(held_values, dtype=np.float32)
            observation, reward, terminated, truncated, info = self.env.step(
                held_action
            )
            total_reward += float(reward)
            substeps += 1
            if terminated or truncated:
                break
        result_info = dict(info)
        result_info.update(
            {
                "ppo_version": self.ppo_config.version,
                "policy_stock_action": float(np.asarray(action).reshape(-1)[0]),
                "policy_futures_action": float(np.asarray(action).reshape(-1)[1]),
                "decision_stock_target": float(target[0]),
                "decision_futures_target": float(target[1]),
                "decision_option_coverages": [float(value) for value in target[2:5]],
                "decision_substeps": substeps,
            }
        )
        self._last_info = result_info
        return observation, total_reward, terminated, truncated, result_info


def wrap_ppo_env(env: gym.Env, config: PPOConfig) -> gym.Env:
    if config.version == "v1":
        return env
    return PPOPortfolioActionWrapper(env, config)
