import pytest

from gas_portfolio_rl.baselines.runner import StrategyRun
from gas_portfolio_rl.evaluation.metrics import episode_metrics
from gas_portfolio_rl.evaluation.splits import SeedManifest


def _row(value: float, *, step: int) -> dict[str, object]:
    return {
        "portfolio_value": value,
        "cumulative_transaction_cost": 10.0 * step,
        "turnover": 0.1,
        "cumulative_dividends": 5.0 * step,
        "margin_calls": 0,
        "termination_reason": None,
        "leverage": 0.5,
        "futures_exposure": -0.2,
        "cash_weight": 0.3,
        "stock_weight": 0.5,
    }


def test_episode_metrics_use_full_equity_path() -> None:
    run = StrategyRun(
        strategy_name="example",
        trajectory=(_row(100.0, step=0), _row(110.0, step=1), _row(99.0, step=2)),
        total_reward=-0.01,
        terminated=False,
        truncated=True,
    )
    result = episode_metrics(
        run, initial_value=100.0, trading_days=2, annual_cash_rate=0.0
    )
    assert result["total_return"] == pytest.approx(-0.01)
    assert result["max_drawdown"] == pytest.approx(0.10)
    assert result["net_pnl"] == pytest.approx(-1.0)
    assert result["transaction_cost"] == 20.0
    assert result["dividends"] == 10.0
    assert result["average_futures_exposure"] == pytest.approx(-0.2)
    assert result["long_futures_fraction"] == 0.0
    assert result["short_futures_fraction"] == 1.0
    assert result["average_cash_weight"] == pytest.approx(0.3)


def test_seed_manifest_ranges_do_not_overlap() -> None:
    manifest = SeedManifest()
    manifest.validate()
    assert manifest.as_dict()["validation"] == {"start": 10_000, "end": 19_999}
