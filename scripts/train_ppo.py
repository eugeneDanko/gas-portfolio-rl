from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.config import resolve_calibrated_market
from gas_portfolio_rl.training import train_ppo


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on calibrated gas portfolio env")
    parser.add_argument("--market-config", type=Path, default=None)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=ROOT / "configs/models/ppo_v5_full.yaml",
    )
    parser.add_argument(
        "--reward-config",
        type=Path,
        default=ROOT / "configs/reward/default.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--training-seed", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="Update the notebook-style diagnostics at validation time")
    args = parser.parse_args()
    market_config = args.market_config or resolve_calibrated_market(ROOT)
    config = AppConfig.from_yaml(
        market_config,
        ROOT / "configs/environment/default.yaml",
        args.reward_config,
        ROOT / "configs/baselines/default.yaml",
        ROOT / "configs/evaluation/default.yaml",
        args.model_config,
    )
    if args.smoke:
        config = replace(
            config,
            ppo=replace(
                config.ppo,
                total_timesteps=4_096,
                n_envs=2,
                n_steps=128,
                batch_size=128,
                n_epochs=2,
                eval_frequency=2_048,
                validation_episodes=2,
                validation_stress_episodes=1,
                checkpoint_frequency=2_048,
            ),
        )
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output = args.output or ROOT / "results/ppo" / config.ppo.experiment_tag / run_name
    result = train_ppo(
        config,
        output,
        training_seed=args.training_seed,
        live_display=args.live,
    )
    print(f"Best validation score: {result.best_validation_score:.6f}")
    print(f"Best model: {result.best_model_path}")
    print(f"Normalization: {result.best_normalization_path}")


if __name__ == "__main__":
    main()
