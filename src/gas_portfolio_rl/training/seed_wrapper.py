from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class EpisodeSeedWrapper(gym.Wrapper):
    """Choose every episode seed from one immutable split."""

    def __init__(
        self,
        env: gym.Env,
        *,
        seed_start: int,
        seed_end: int,
        selector_seed: int,
    ) -> None:
        super().__init__(env)
        if seed_start < 0 or seed_end < seed_start:
            raise ValueError("Invalid episode seed range")
        self.seed_start = seed_start
        self.seed_end = seed_end
        self.selector_seed = selector_seed
        self._selector = np.random.default_rng(selector_seed)
        self.last_episode_seed: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self._selector = np.random.default_rng(seed)
        episode_seed = int(self._selector.integers(self.seed_start, self.seed_end + 1))
        self.last_episode_seed = episode_seed
        observation, info = self.env.reset(seed=episode_seed, options=options)
        info = dict(info)
        info["episode_seed"] = episode_seed
        return observation, info
