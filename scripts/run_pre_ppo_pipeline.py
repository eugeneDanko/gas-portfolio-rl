from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.calibration import calibrate_market, save_calibration
from gas_portfolio_rl.evaluation import (
    build_reward_gate,
    evaluate_strategies,
    make_default_strategies,
    make_ppo_feasible_grid_strategies,
    make_scenarios,
    save_reward_gate,
)
from gas_portfolio_rl.evaluation.reporting import save_evaluation
from gas_portfolio_rl.evaluation.preflight import run_preflight_checks
from gas_portfolio_rl.evaluation.splits import SeedManifest


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate before PPO training")
    parser.add_argument("--quick", action="store_true", help="Use tiny samples for a smoke run")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()
    config = AppConfig.from_yaml(
        ROOT / "configs/market/default.yaml",
        ROOT / "configs/environment/default.yaml",
        ROOT / "configs/reward/default.yaml",
        ROOT / "configs/baselines/default.yaml",
        ROOT / "configs/calibration/default.yaml",
        ROOT / "configs/evaluation/default.yaml",
        ROOT / "configs/models/ppo.yaml",
    )
    if args.quick:
        config = replace(
            config,
            calibration=replace(config.calibration, paths_per_candidate=2,
                                candidate_count=2, path_steps=60),
            evaluation=replace(config.evaluation, seeds=3, stress_seeds=2,
                               reward_grid_seeds=2, bootstrap_samples=30),
        )

    output = ROOT / "results/pre_ppo"
    output.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "quick_run": args.quick,
        "calibration_enabled": not args.skip_calibration,
        "calibration_paths_per_candidate": config.calibration.paths_per_candidate,
        "calibration_candidate_count": config.calibration.candidate_count,
        "validation_seeds": config.evaluation.seeds,
        "stress_seeds": config.evaluation.stress_seeds,
        "final_evaluation": not args.quick,
    }
    with (output / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2)
    manifest = SeedManifest()
    with (output / "seed_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest.as_dict(), stream, indent=2)
    with (output / "preflight.json").open("w", encoding="utf-8") as stream:
        json.dump(run_preflight_checks(config), stream, indent=2)

    if not args.skip_calibration:
        calibration = calibrate_market(config.market, config.calibration)
        if not args.quick and not calibration.passes_gate:
            raise RuntimeError(
                f"Calibration gate failed: objective={calibration.objective:.4f}, "
                f"maximum={calibration.maximum_objective:.4f}, "
                f"individual_moments_pass={calibration.individual_moments_pass}, "
                f"errors={calibration.errors.as_dict()}"
            )
        config = replace(config, market=calibration.market_config)
        save_calibration(calibration, output)
    else:
        calibrated_path = output / "calibrated_market.yaml"
        report_path = output / "calibration_report.yaml"
        if not calibrated_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(
                "--skip-calibration requires existing calibrated_market.yaml "
                "and calibration_report.yaml in results/pre_ppo"
            )
        calibrated = AppConfig.from_yaml(calibrated_path)
        config = replace(config, market=calibrated.market)

    scenarios = make_scenarios(config)
    validation = list(range(config.evaluation.seed_start,
                            config.evaluation.seed_start + config.evaluation.seeds))
    stress = list(range(30_000, 30_000 + config.evaluation.stress_seeds))
    seeds = {"base": validation, "calm": validation, "crisis_stress": stress}
    result = evaluate_strategies(
        scenarios,
        make_default_strategies(config),
        seeds=seeds,
    )
    save_evaluation(result, output)
    grid_count = config.evaluation.reward_grid_seeds
    grid_seeds = {
        "base": validation[:grid_count],
        "calm": validation[:grid_count],
        "crisis_stress": stress[:grid_count],
    }
    grid_result = evaluate_strategies(
        scenarios,
        make_ppo_feasible_grid_strategies(config),
        seeds=grid_seeds,
    )
    save_evaluation(grid_result, output / "reward_grid")
    reward_gate, ranking = build_reward_gate(result, grid_result, config)
    save_reward_gate(reward_gate, ranking, output)
    print(f"Saved pre-PPO artifacts to {output}")
    if args.quick:
        print("SMOKE RUN ONLY: do not treat these estimates as final statistics")
    for row in result.summary:
        print(row["scenario"], row["strategy"],
              f"return={row['total_return_mean']:.3%}",
              f"maxDD={row['max_drawdown_mean']:.3%}",
              f"reward={row['total_reward_mean']:.3f}")
    print("Reward gate:", reward_gate)


if __name__ == "__main__":
    main()
