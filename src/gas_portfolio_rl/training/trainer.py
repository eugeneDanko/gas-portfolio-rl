from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any

from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.training.env_factory import make_training_vec_env


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    final_model_path: Path
    final_normalization_path: Path
    best_model_path: Path
    best_normalization_path: Path
    best_validation_score: float


def train_ppo(
    config: AppConfig,
    output_dir: str | Path,
    *,
    training_seed: int = 1_000,
    train_seed_start: int = 1_000,
    train_seed_end: int = 9_999,
    validation_seed_start: int = 10_000,
    stress_validation_seed_start: int = 30_000,
    live_display: bool = False,
) -> TrainingResult:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed

    from gas_portfolio_rl.training.callbacks import RiskValidationCallback

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    set_random_seed(training_seed)
    vector_env = make_training_vec_env(
        config,
        monitor_dir=destination / "monitor",
        selector_seed=training_seed,
        seed_start=train_seed_start,
        seed_end=train_seed_end,
    )
    tensorboard_available = importlib.util.find_spec("tensorboard") is not None
    model = PPO(
        "MlpPolicy",
        vector_env,
        learning_rate=(
            (lambda progress_remaining: config.ppo.learning_rate * progress_remaining)
            if config.ppo.learning_rate_schedule == "linear"
            else config.ppo.learning_rate
        ),
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        n_epochs=config.ppo.n_epochs,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        ent_coef=config.ppo.ent_coef,
        vf_coef=config.ppo.vf_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        policy_kwargs={
            "net_arch": {
                "pi": list(config.ppo.policy_layers),
                "vf": list(config.ppo.policy_layers),
            }
        },
        tensorboard_log=str(destination / "tensorboard") if tensorboard_available else None,
        seed=training_seed,
        device=config.ppo.device,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.ppo.checkpoint_frequency // config.ppo.n_envs, 1),
        save_path=str(checkpoints),
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    validation_callback = RiskValidationCallback(
        config,
        destination,
        seed_start=validation_seed_start,
        stress_seed_start=stress_validation_seed_start,
        live_display=live_display,
    )
    metadata: dict[str, Any] = {
        "algorithm": "PPO",
        "training_seed": training_seed,
        "train_seed_range": [train_seed_start, train_seed_end],
        "validation_seed_start": validation_seed_start,
        "stress_validation_seed_start": stress_validation_seed_start,
        "test_seeds_used": False,
        "tensorboard_enabled": tensorboard_available,
        "config": asdict(config),
    }
    with (destination / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    try:
        model.learn(
            total_timesteps=config.ppo.total_timesteps,
            callback=CallbackList([checkpoint_callback, validation_callback]),
            progress_bar=False,
        )
        model.save(destination / "final_model")
        vector_env.save(destination / "final_vecnormalize.pkl")
    finally:
        vector_env.close()

    return TrainingResult(
        output_dir=destination,
        final_model_path=destination / "final_model.zip",
        final_normalization_path=destination / "final_vecnormalize.pkl",
        best_model_path=destination / "best_model.zip",
        best_normalization_path=destination / "best_vecnormalize.pkl",
        best_validation_score=float(validation_callback.best_score),
    )
