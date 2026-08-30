from pathlib import Path

import pytest

from gas_portfolio_rl.config import resolve_calibrated_market


def _write_candidate(
    root: Path,
    directory: str,
    *,
    overall_pass: bool,
    moments_pass: bool,
    reward_pass: bool | None = None,
) -> Path:
    destination = root / "results" / directory
    destination.mkdir(parents=True)
    market = destination / "calibrated_market.yaml"
    market.write_text("market:\n  trading_days: 252\n", encoding="utf-8")
    (destination / "calibration_report.yaml").write_text(
        "passes_calibration_gate: " + str(overall_pass).lower() + "\n"
        "individual_moments_pass: " + str(moments_pass).lower() + "\n",
        encoding="utf-8",
    )
    if reward_pass is not None:
        (destination / "reward_gate.json").write_text(
            '{"passes_reward_gate": ' + str(reward_pass).lower() + "}\n",
            encoding="utf-8",
        )
    return market


def test_resolver_skips_failed_quick_run_and_uses_accepted_calibration(
    tmp_path: Path,
) -> None:
    _write_candidate(
        tmp_path,
        "pre_ppo",
        overall_pass=False,
        moments_pass=False,
    )
    accepted = _write_candidate(
        tmp_path,
        "calibration",
        overall_pass=True,
        moments_pass=True,
    )

    assert resolve_calibrated_market(tmp_path) == accepted


def test_resolver_rejects_all_uncalibrated_candidates(tmp_path: Path) -> None:
    _write_candidate(
        tmp_path,
        "pre_ppo",
        overall_pass=True,
        moments_pass=False,
    )

    with pytest.raises(FileNotFoundError, match="No accepted calibrated market"):
        resolve_calibrated_market(tmp_path)


def test_resolver_requires_passed_reward_gate_for_pre_ppo(tmp_path: Path) -> None:
    _write_candidate(
        tmp_path,
        "pre_ppo",
        overall_pass=True,
        moments_pass=True,
        reward_pass=False,
    )

    with pytest.raises(FileNotFoundError, match="reward gate failed"):
        resolve_calibrated_market(tmp_path)


def test_resolver_accepts_full_pre_ppo_gates(tmp_path: Path) -> None:
    accepted = _write_candidate(
        tmp_path,
        "pre_ppo",
        overall_pass=True,
        moments_pass=True,
        reward_pass=True,
    )

    assert resolve_calibrated_market(tmp_path) == accepted
