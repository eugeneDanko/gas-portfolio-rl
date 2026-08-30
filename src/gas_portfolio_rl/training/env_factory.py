from __future__ import annotations

from pathlib import Path

from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv
from gas_portfolio_rl.evaluation.scenarios import make_scenarios
from gas_portfolio_rl.training.seed_wrapper import EpisodeSeedWrapper
from gas_portfolio_rl.training.action_wrapper import wrap_ppo_env


def make_training_vec_env(
    config: AppConfig,
    *,
    monitor_dir: str | Path,
    selector_seed: int,
    seed_start: int = 1_000,
    seed_end: int = 9_999,
):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    destination = Path(monitor_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stress_config = make_scenarios(config)["crisis_stress"]
    stress_env_count = (
        int(round(config.ppo.n_envs * config.ppo.stress_training_fraction))
        if config.ppo.version in {"v3", "v4"}
        else 0
    )
    stress_rank_start = config.ppo.n_envs - stress_env_count

    def factory(rank: int):
        def create():
            episode_config = stress_config if rank >= stress_rank_start else config
            env = EpisodeSeedWrapper(
                GasPortfolioEnv(episode_config),
                seed_start=seed_start,
                seed_end=seed_end,
                selector_seed=selector_seed + rank,
            )
            env = wrap_ppo_env(env, episode_config.ppo)
            return Monitor(env, filename=str(destination / f"env_{rank}"))

        return create

    vector_env = DummyVecEnv([factory(rank) for rank in range(config.ppo.n_envs)])
    vector_env.seed(selector_seed)
    return VecNormalize(
        vector_env,
        norm_obs=config.ppo.normalize_observations,
        norm_reward=config.ppo.normalize_rewards,
        clip_obs=config.ppo.clip_observation,
        clip_reward=config.ppo.clip_reward,
        gamma=config.ppo.gamma,
    )
