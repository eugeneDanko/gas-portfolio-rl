from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from gas_portfolio_rl.calibration.market import CalibrationResult


def save_calibration(result: CalibrationResult, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": result.objective,
        "candidates_evaluated": result.candidates_evaluated,
        "maximum_objective": result.maximum_objective,
        "passes_calibration_gate": result.passes_gate,
        "individual_moments_pass": result.individual_moments_pass,
        "moment_errors": result.errors.as_dict(),
        "targets": result.targets.as_dict(),
        "tolerances": result.tolerances.as_dict(),
        "moments": result.moments.as_dict(),
        "selected_parameters": {
            "transition_matrix": result.market_config.transition_matrix,
            "normal": asdict(result.market_config.normal),
            "crisis": asdict(result.market_config.crisis),
        },
    }
    with (destination / "calibration_report.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)

    calibrated_market = asdict(result.market_config)
    calibrated_market["seasonality_anchors"] = [
        list(item) for item in result.market_config.seasonality_anchors
    ]
    calibrated_market["transition_matrix"] = [
        list(row) for row in result.market_config.transition_matrix
    ]
    with (destination / "calibrated_market.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"market": calibrated_market}, stream, sort_keys=False)
