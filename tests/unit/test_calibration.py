from dataclasses import replace

from gas_portfolio_rl import AppConfig
from gas_portfolio_rl.calibration import calibrate_market, estimate_market_moments, save_calibration
from gas_portfolio_rl.config.models import CalibrationConfig, MarketConfig


def test_market_moments_are_seed_reproducible() -> None:
    config = MarketConfig(episode_steps=40)
    first = estimate_market_moments(config, seeds=[1, 2], steps=40)
    second = estimate_market_moments(config, seeds=[1, 2], steps=40)
    assert first == second
    assert first.gas_volatility > 0
    assert -1 <= first.stock_futures_correlation <= 1


def test_calibration_returns_one_of_requested_candidates() -> None:
    settings = CalibrationConfig(paths_per_candidate=2, candidate_count=3, path_steps=30)
    result = calibrate_market(MarketConfig(), settings)
    assert result.candidates_evaluated == 3
    assert result.objective >= 0
    assert result.market_config.normal.gas_volatility > 0
    assert result.passes_gate == (
        result.objective <= settings.maximum_objective
        and result.individual_moments_pass
    )
    assert set(result.errors.as_dict()) == {
        "gas_volatility_relative",
        "stock_volatility_relative",
        "futures_volatility_relative",
        "stock_futures_correlation_absolute",
        "crisis_fraction_absolute",
    }


def test_saved_calibrated_market_is_loadable(tmp_path) -> None:
    settings = CalibrationConfig(paths_per_candidate=1, candidate_count=1, path_steps=20)
    result = calibrate_market(MarketConfig(), settings)
    save_calibration(result, tmp_path)
    loaded = AppConfig.from_yaml(tmp_path / "calibrated_market.yaml")
    assert loaded.market == result.market_config
    assert (tmp_path / "calibration_report.yaml").exists()
