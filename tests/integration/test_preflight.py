from dataclasses import replace

from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.evaluation.preflight import run_preflight_checks


def test_preflight_checks_environment_contract_and_determinism() -> None:
    config = AppConfig()
    config = replace(config, market=replace(config.market, episode_steps=20))
    result = run_preflight_checks(config, steps=10)
    assert result["gymnasium_check"]
    assert result["seed_determinism"]
    assert result["finite_observations_and_rewards"]
    assert result["observation_shape"] == [23]
    assert result["action_shape"] == [2]
