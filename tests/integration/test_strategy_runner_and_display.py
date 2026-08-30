from dataclasses import replace

from gas_portfolio_rl import AppConfig, GasPortfolioEnv, RelativeMomentumConfig
from gas_portfolio_rl.baselines import RelativeMomentumBaseline, run_strategy
from gas_portfolio_rl.visualization import LiveTradingDisplay


def test_strategy_runner_and_display_create_complete_artifact(tmp_path, monkeypatch):
    base = AppConfig()
    config = replace(base, market=replace(base.market, episode_steps=4))
    strategy = RelativeMomentumBaseline(
        RelativeMomentumConfig(lookback=2, rebalance_interval=1),
    )
    monitor = LiveTradingDisplay(live=True, refresh_every=1, window_size=3)
    live_figure_ids = []
    monkeypatch.setattr(
        monitor,
        "_show_live",
        lambda: live_figure_ids.append(id(monitor._live_figure)),
    )

    result = run_strategy(GasPortfolioEnv(config), strategy, seed=7, monitor=monitor)
    output = monitor.save(tmp_path / "monitor.png")

    assert result.truncated and not result.terminated
    assert len(result.trajectory) == 5
    assert monitor.trades
    assert len(set(live_figure_ids)) == 1
    live_figure = monitor.render(full_history=False)
    assert list(live_figure.axes[0].lines[0].get_xdata()) == [2, 3, 4]
    assert live_figure.axes[0].get_xlim() == (2.0, 4.0)
    assert output.exists() and output.stat().st_size > 0
