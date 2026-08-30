from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from gas_portfolio_rl.simulation.models import MarketState, Regime


def plot_market_prices(
    states: Sequence[MarketState],
    output_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot gas spot/front future and GasCo, with crisis periods shaded."""
    if not states:
        raise ValueError("states must not be empty")
    steps = np.arange(len(states))
    gas = np.asarray([state.gas_spot for state in states])
    futures = np.asarray([state.futures_price for state in states])
    stock = np.asarray([state.stock_price for state in states])

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(steps, gas, label="Gas spot", linewidth=1.5)
    axes[0].plot(steps, futures, label="Front futures", linewidth=1.3)
    axes[0].set_ylabel("Price, synthetic units")
    axes[0].set_title("Synthetic gas market")
    axes[0].legend(loc="best")

    axes[1].plot(steps, stock, label="GasCo", color="tab:green", linewidth=1.5)
    axes[1].set_ylabel("Stock price")
    axes[1].set_xlabel("Trading step")
    axes[1].set_title("GasCo stock")
    axes[1].legend(loc="best")

    for axis in axes:
        _shade_crisis_periods(axis, states)
        axis.grid(alpha=0.25)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, bbox_inches="tight")
    return figure, axes


def plot_forward_curve(
    state: MarketState,
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot the observable term structure for one valuation date."""
    if not state.forward_curve or not state.forward_delivery_dates:
        raise ValueError("state does not contain a forward curve")
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(
        state.forward_delivery_dates,
        state.forward_curve,
        marker="o",
        linewidth=1.6,
        label="Synthetic gas forward curve",
    )
    valuation = state.calendar_date.isoformat() if state.calendar_date else f"step {state.day}"
    axis.set_title(f"Forward curve observed on {valuation}")
    axis.set_xlabel("Contract delivery date")
    axis.set_ylabel("Futures price, synthetic units")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.autofmt_xdate()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, bbox_inches="tight")
    return figure, axis


def _shade_crisis_periods(axis: Axes, states: Sequence[MarketState]) -> None:
    in_crisis = False
    start = 0
    label_used = False
    for index, state in enumerate(states):
        crisis = state.regime is Regime.CRISIS
        if crisis and not in_crisis:
            start = index
            in_crisis = True
        if in_crisis and (not crisis or index == len(states) - 1):
            end = index if not crisis else index + 1
            axis.axvspan(
                start,
                end,
                color="tab:red",
                alpha=0.10,
                label="Crisis" if not label_used else None,
            )
            label_used = True
            in_crisis = False
