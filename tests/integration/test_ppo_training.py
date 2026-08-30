from dataclasses import replace
import json

import pytest

pytest.importorskip("stable_baselines3")

from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.evaluation.policy import load_policy_bundle
from gas_portfolio_rl.training import train_ppo


def test_ppo_smoke_training_saves_complete_bundle(tmp_path) -> None:
    base = AppConfig()
    config = replace(
        base,
        market=replace(base.market, episode_steps=40),
        ppo=replace(
            base.ppo,
            total_timesteps=256,
            n_envs=1,
            n_steps=64,
            batch_size=64,
            n_epochs=1,
            eval_frequency=128,
            validation_episodes=1,
            validation_stress_episodes=1,
            checkpoint_frequency=128,
        ),
    )
    result = train_ppo(config, tmp_path, training_seed=1_000)
    assert result.best_model_path.exists()
    assert result.best_normalization_path.exists()
    assert result.final_model_path.exists()
    assert result.final_normalization_path.exists()
    assert (tmp_path / "validation_metrics.csv").exists()
    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["train_seed_range"] == [1_000, 9_999]
    assert metadata["test_seeds_used"] is False
    model, normalizer = load_policy_bundle(
        str(result.best_model_path), str(result.best_normalization_path), config
    )
    assert model is not None
    normalizer.close()
