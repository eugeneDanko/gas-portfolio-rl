from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv


class Strategy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray: ...


class StrategyMonitor(Protocol):
    def reset(self, strategy_name: str, info: Mapping[str, Any]) -> None: ...

    def update(
        self,
        strategy_name: str,
        info: Mapping[str, Any],
        action: np.ndarray,
        done: bool,
    ) -> None: ...


@dataclass(frozen=True)
class StrategyRun:
    strategy_name: str
    trajectory: tuple[dict[str, Any], ...]
    total_reward: float
    terminated: bool
    truncated: bool


class ModelStrategy:
    """Adapter for Stable-Baselines3-like models exposing ``predict``."""

    def __init__(self, model: Any, name: str = "rl_model", deterministic: bool = True) -> None:
        self.model = model
        self.name = name
        self.deterministic = deterministic

    def reset(self) -> None:
        return None

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> np.ndarray:
        del info
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)


def run_strategy(
    env: GasPortfolioEnv,
    strategy: Strategy,
    *,
    seed: int = 42,
    max_steps: int | None = None,
    monitor: StrategyMonitor | None = None,
) -> StrategyRun:
    observation, info = env.reset(seed=seed)
    strategy.reset()
    if monitor is not None:
        monitor.reset(strategy.name, info)

    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        action = strategy.act(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if max_steps is not None and steps >= max_steps and not terminated:
            truncated = True
        if monitor is not None:
            monitor.update(strategy.name, info, action, terminated or truncated)

    return StrategyRun(
        strategy_name=strategy.name,
        trajectory=tuple(dict(row) for row in env.trajectory),
        total_reward=float(total_reward),
        terminated=terminated,
        truncated=truncated,
    )
