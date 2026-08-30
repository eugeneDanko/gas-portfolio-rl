from __future__ import annotations

import argparse
from pathlib import Path

from gas_portfolio_rl import AppConfig, GasPortfolioEnv
from gas_portfolio_rl.baselines import (
    RelativeMomentumBaseline,
    RollingHedgeBaseline,
    SignedRollingHedgeBaseline,
    run_strategy,
)
from gas_portfolio_rl.visualization import LiveTradingDisplay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline with a live trading display")
    parser.add_argument(
        "--strategy",
        choices=("momentum", "hedge", "signed-hedge"),
        default="momentum",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--refresh-every", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=80)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = AppConfig.from_yaml(
        project_root / "configs" / "market" / "default.yaml",
        project_root / "configs" / "environment" / "default.yaml",
        project_root / "configs" / "reward" / "default.yaml",
        project_root / "configs" / "baselines" / "default.yaml",
    )
    if args.strategy == "momentum":
        strategy = RelativeMomentumBaseline(
            config.baselines.relative_momentum,
            trading_days=config.market.trading_days,
        )
    elif args.strategy == "hedge":
        strategy = RollingHedgeBaseline(config.baselines.rolling_hedge)
    else:
        strategy = SignedRollingHedgeBaseline(config.baselines.signed_rolling_hedge)

    monitor = LiveTradingDisplay(
        refresh_every=args.refresh_every,
        window_size=args.window_size,
        pause_seconds=args.pause_seconds,
        live=not args.no_live,
    )
    result = run_strategy(
        GasPortfolioEnv(config),
        strategy,
        seed=args.seed,
        max_steps=args.steps,
        monitor=monitor,
    )
    output = args.output or (
        project_root / "results" / "reports" / f"{result.strategy_name}_live_monitor.png"
    )
    monitor.save(output)
    final = result.trajectory[-1]
    print(
        f"strategy={result.strategy_name} steps={final['step']} "
        f"value={final['portfolio_value']:.2f} pnl={final['pnl']:.2f} "
        f"reward={result.total_reward:.4f} output={output}"
    )


if __name__ == "__main__":
    main()
