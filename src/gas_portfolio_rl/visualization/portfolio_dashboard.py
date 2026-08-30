from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_portfolio_dashboard(
    trajectory: Sequence[Mapping[str, Any]],
    output_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Static dashboard fed directly by ``GasPortfolioEnv.trajectory``."""
    if not trajectory:
        raise ValueError("trajectory must not be empty")
    step = np.asarray([row["step"] for row in trajectory])
    value = np.asarray([row["portfolio_value"] for row in trajectory])
    cash = np.asarray([row["cash"] for row in trajectory])
    pnl = np.asarray([row["pnl"] for row in trajectory])
    leverage = np.asarray([row["leverage"] for row in trajectory])
    stock = np.asarray([row["stock_weight"] for row in trajectory])
    futures = np.asarray([row["futures_exposure"] for row in trajectory])
    cash_weight = np.asarray([row["cash_weight"] for row in trajectory])

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0, 0].plot(step, value, label="Portfolio value")
    axes[0, 0].plot(step, cash, label="Cash", alpha=0.8)
    axes[0, 0].set_ylabel("Money")
    axes[0, 0].set_title("Capital and cash")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(step, pnl, color="tab:blue")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("Money")
    axes[0, 1].set_title("Cumulative P&L")

    axes[1, 0].plot(step, leverage, color="tab:orange")
    axes[1, 0].set_ylabel("Gross exposure / equity")
    axes[1, 0].set_xlabel("Trading step")
    axes[1, 0].set_title("Leverage")

    axes[1, 1].plot(step, stock, label="Stock weight")
    axes[1, 1].plot(step, futures, label="Futures exposure")
    axes[1, 1].plot(step, cash_weight, label="Cash weight")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("Share / exposure")
    axes[1, 1].set_xlabel("Trading step")
    axes[1, 1].set_title("Allocation")
    axes[1, 1].legend(loc="best")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, bbox_inches="tight")
    return figure, axes

