from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gas_portfolio_rl.config.models import AppConfig, MarketConfig  # noqa: E402
from gas_portfolio_rl.simulation.market_simulator import MarketSimulator  # noqa: E402
from gas_portfolio_rl.visualization.prices import plot_forward_curve, plot_market_prices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a seeded synthetic market trajectory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=504)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "market" / "default.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "reports" / "market_prices.png",
    )
    parser.add_argument(
        "--curve-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "reports" / "forward_curve.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    market_config: MarketConfig = AppConfig.from_yaml(args.config).market
    states = MarketSimulator(market_config).simulate(args.steps, seed=args.seed)
    figure, _ = plot_market_prices(states, args.output)
    figure.clear()
    curve_figure, _ = plot_forward_curve(states[0], args.curve_output)
    curve_figure.clear()
    print(f"Saved {args.output.resolve()}")
    print(f"Saved {args.curve_output.resolve()}")


if __name__ == "__main__":
    main()
