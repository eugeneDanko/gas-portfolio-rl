from __future__ import annotations

import argparse
import json
from pathlib import Path

from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.config import resolve_calibrated_market
from gas_portfolio_rl.evaluation.ppo import evaluate_ppo_bundle
from gas_portfolio_rl.evaluation.acceptance import assess_ppo
from gas_portfolio_rl.evaluation.ppo_registry import (
    load_ppo_registry,
    register_ppo_evaluation,
)
from gas_portfolio_rl.evaluation.reporting import save_evaluation
from gas_portfolio_rl.evaluation.scenarios import make_scenarios
from gas_portfolio_rl.visualization import (
    plot_evaluation_diagnostics,
    plot_versioned_pnl_comparison,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO against all baselines")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--market-config", type=Path, default=None)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "results/ppo/registry/ppo_history.csv",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--test", action="store_true",
                        help="Use untouched test seeds; only for the final selected model")
    args = parser.parse_args()
    metadata_path = args.run_dir / "run_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = AppConfig.from_mapping(metadata["config"])
    else:
        market_config = args.market_config or resolve_calibrated_market(ROOT)
        config = AppConfig.from_yaml(
            market_config,
            ROOT / "configs/environment/default.yaml",
            ROOT / "configs/reward/default.yaml",
            ROOT / "configs/baselines/default.yaml",
            ROOT / "configs/evaluation/default.yaml",
            ROOT / "configs/models/ppo.yaml",
        )
    count = 3 if args.quick else config.evaluation.seeds
    stress_count = 2 if args.quick else config.evaluation.stress_seeds
    start = 20_000 if args.test else config.evaluation.seed_start
    stress_start = 40_000 if args.test else 30_000
    scenarios = make_scenarios(config)
    seeds = {
        "base": range(start, start + count),
        "calm": range(start, start + count),
        "crisis_stress": range(stress_start, stress_start + stress_count),
    }
    result = evaluate_ppo_bundle(
        config,
        scenarios,
        model_path=args.run_dir / "best_model.zip",
        normalization_path=args.run_dir / "best_vecnormalize.pkl",
        seeds=seeds,
    )
    output = args.run_dir / ("test_evaluation" if args.test else "validation_evaluation")
    save_evaluation(result, output)
    acceptance = assess_ppo(result, config)
    with (output / "acceptance.json").open("w", encoding="utf-8") as stream:
        json.dump(acceptance, stream, indent=2)
    split = "test" if args.test else "validation"
    registry = register_ppo_evaluation(
        result,
        run_dir=args.run_dir,
        config=config,
        acceptance=acceptance,
        registry_path=args.registry,
        evaluation_split=split,
    )
    diagnostics = plot_evaluation_diagnostics(
        result.summary,
        initial_value=config.portfolio.initial_cash,
    )
    diagnostics.savefig(output / "evaluation_diagnostics.png", dpi=150)
    versioned_pnl = plot_versioned_pnl_comparison(
        result.summary,
        registry,
        initial_value=config.portfolio.initial_cash,
        evaluation_split=split,
    )
    versioned_pnl.savefig(output / "versioned_net_pnl.png", dpi=150)
    for row in result.summary:
        print(row["scenario"], row["strategy"],
              f"return={row['total_return_mean']:.2%}",
              f"DD={row['max_drawdown_mean']:.2%}",
              f"reward={row['total_reward_mean']:.3f}")
    print("PPO acceptance:", acceptance)


if __name__ == "__main__":
    main()
