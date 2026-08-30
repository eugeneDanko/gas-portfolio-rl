from __future__ import annotations

from pathlib import Path

import json
import yaml


def resolve_calibrated_market(project_root: str | Path) -> Path:
    """Return only a calibrated market whose persisted gates both passed."""
    root = Path(project_root).resolve()
    candidates = (
        root / "results" / "pre_ppo" / "calibrated_market.yaml",
        root / "results" / "calibration" / "calibrated_market.yaml",
    )
    failures: list[str] = []
    for market_path in candidates:
        report_path = market_path.with_name("calibration_report.yaml")
        if not market_path.is_file() or not report_path.is_file():
            failures.append(f"missing {market_path} or {report_path}")
            continue
        with report_path.open("r", encoding="utf-8") as stream:
            report = yaml.safe_load(stream) or {}
        calibration_passed = bool(report.get("passes_calibration_gate")) and bool(
            report.get("individual_moments_pass")
        )
        if calibration_passed and market_path.parent.name == "pre_ppo":
            reward_gate_path = market_path.with_name("reward_gate.json")
            if not reward_gate_path.is_file():
                raise FileNotFoundError(
                    f"Accepted calibration has no reward gate: {reward_gate_path}"
                )
            with reward_gate_path.open("r", encoding="utf-8") as stream:
                reward_gate = json.load(stream)
            if not bool(reward_gate.get("passes_reward_gate")):
                raise FileNotFoundError(
                    f"Pre-PPO reward gate failed in {reward_gate_path}"
                )
            return market_path
        if calibration_passed:
            return market_path
        failures.append(f"failed calibration gates in {report_path}")
    raise FileNotFoundError(
        "No accepted calibrated market. Run 02_calibration_and_evaluation.ipynb "
        f"with QUICK_RUN=False. Checked: {'; '.join(failures)}"
    )
