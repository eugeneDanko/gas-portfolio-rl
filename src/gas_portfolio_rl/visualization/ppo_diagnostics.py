from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        lines = [line for line in stream if not line.startswith("#")]
    return list(csv.DictReader(lines))


def _rolling_mean(values: Iterable[float], window: int = 50) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return array
    result = np.empty_like(array)
    cumulative = np.cumsum(array)
    for index in range(array.size):
        start = max(0, index - window + 1)
        total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
        result[index] = total / (index - start + 1)
    return result


def plot_training_diagnostics(run_dir: str | Path, *, save: bool = True) -> Figure:
    """Plot rollout and risk-aware validation dynamics for one PPO run."""
    root = Path(run_dir)
    monitor_rows: list[dict[str, str]] = []
    for path in sorted((root / "monitor").glob("*.monitor.csv")):
        monitor_rows.extend(_read_csv(path))
    monitor_rows.sort(key=lambda row: float(row["t"]))
    validation = _read_csv(root / "validation_metrics.csv")

    figure = Figure(figsize=(15, 12))
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    figure.suptitle(f"PPO training diagnostics — {root.name}", fontsize=14)

    rewards = [float(row["r"]) for row in monitor_rows]
    lengths = [float(row["l"]) for row in monitor_rows]
    episode_index = np.arange(1, len(rewards) + 1)
    axes[0, 0].plot(episode_index, _rolling_mean(rewards), color="tab:blue")
    axes[0, 0].set_title("Training episode reward — rolling mean (50)")
    axes[0, 0].set_xlabel("Completed episodes")
    axes[0, 0].set_ylabel("Raw reward")
    axes[0, 1].plot(episode_index, _rolling_mean(lengths), color="tab:orange")
    axes[0, 1].set_title("Training episode length — rolling mean (50)")
    axes[0, 1].set_xlabel("Completed episodes")
    axes[0, 1].set_ylabel("PPO decisions")

    if validation:
        steps = np.asarray([float(row["timesteps"]) for row in validation])
        axes[1, 0].plot(steps, [float(row["score"]) for row in validation], label="score")
        axes[1, 0].plot(steps, [float(row["mean_reward"]) for row in validation], label="mean reward")
        if "crisis_mean_reward" in validation[0]:
            axes[1, 0].plot(
                steps,
                [float(row["crisis_mean_reward"]) for row in validation],
                label="crisis reward",
            )
        axes[1, 0].set_title("Risk-aware validation objective")
        axes[1, 0].legend()
        axes[1, 1].plot(steps, [100 * float(row["mean_total_return"]) for row in validation], label="return, %")
        axes[1, 1].plot(steps, [100 * float(row["mean_max_drawdown"]) for row in validation], label="max drawdown, %")
        if "crisis_mean_total_return" in validation[0]:
            axes[1, 1].plot(
                steps,
                [100 * float(row["crisis_mean_total_return"]) for row in validation],
                label="crisis return, %",
            )
            axes[1, 1].plot(
                steps,
                [100 * float(row["crisis_mean_max_drawdown"]) for row in validation],
                label="crisis drawdown, %",
            )
            if "crisis_put_ladder_mean_max_drawdown" in validation[0]:
                axes[1, 1].plot(
                    steps,
                    [100 * float(row["crisis_put_ladder_mean_max_drawdown"]) for row in validation],
                    label="put-ladder crisis drawdown, %",
                    linestyle="--",
                )
        axes[1, 1].set_title("Validation performance")
        axes[1, 1].legend()
        for key, label in (("margin_call_rate", "margin call"), ("bankruptcy_rate", "bankruptcy"), ("high_cash_fraction", "high cash")):
            axes[2, 0].plot(steps, [100 * float(row[key]) for row in validation], label=label)
        axes[2, 0].set_title("Validation risk and degeneracy rates")
        axes[2, 0].set_ylabel("Fraction, %")
        axes[2, 0].legend()
        axes[2, 1].plot(steps, [100 * float(row["mean_stock_weight"]) for row in validation], label="stock weight, %")
        if "mean_option_coverage" in validation[0]:
            axes[2, 1].plot(
                steps,
                [100 * float(row["mean_option_coverage"]) for row in validation],
                label="put coverage, %",
            )
        axes[2, 1].plot(steps, [float(row["mean_turnover"]) for row in validation], label="turnover")
        axes[2, 1].set_title("Participation and trading activity")
        axes[2, 1].legend()
    else:
        for axis in axes[1:, :].flat:
            axis.text(0.5, 0.5, "Validation has not run yet", ha="center", va="center")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    if save:
        figure.savefig(root / "training_diagnostics.png", dpi=150)
    return figure


def plot_evaluation_diagnostics(
    summary: Iterable[Mapping[str, Any]],
    *,
    initial_value: float = 1_000_000.0,
) -> Figure:
    """Compare PPO to the main insurance baselines across evaluation scenarios."""
    rows = list(summary)
    scenarios = [name for name in ("base", "calm", "crisis_stress") if any(str(row["scenario"]) == name for row in rows)]
    strategies = [
        "cash",
        "minimum_stock_unhedged",
        "minimum_stock_put_ladder",
        "signed_rolling_hedge",
        "ppo",
    ]
    lookup = {(str(row["scenario"]), str(row["strategy"])): row for row in rows}
    present = [strategy for strategy in strategies if all((scenario, strategy) in lookup for scenario in scenarios)]
    all_present = sorted(
        {
            str(row["strategy"])
            for row in rows
            if all((scenario, str(row["strategy"])) in lookup for scenario in scenarios)
        },
        key=lambda strategy: (strategy == "ppo", strategy),
    )
    if not scenarios or "ppo" not in present:
        raise ValueError("Evaluation diagnostics require PPO rows")

    figure = Figure(figsize=(16, 14))
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    x = np.arange(len(scenarios), dtype=np.float64)
    width = 0.8 / max(len(present), 1)
    colors = {
        "cash": "tab:gray",
        "minimum_stock_unhedged": "tab:orange",
        "minimum_stock_put_ladder": "tab:red",
        "signed_rolling_hedge": "tab:green",
        "ppo": "tab:blue",
    }
    metrics = (
        (axes[0, 0], "total_return_mean", 100.0, "Mean total return", "Return, %"),
        (axes[0, 1], "max_drawdown_mean", 100.0, "Mean maximum drawdown", "Drawdown, %"),
        (axes[1, 0], "total_reward_mean", 1.0, "Mean environment reward", "Reward"),
    )
    for index, strategy in enumerate(present):
        offset = (index - (len(present) - 1) / 2) * width
        for axis, metric, scale, title, ylabel in metrics:
            values = []
            for scenario in scenarios:
                row = lookup[(scenario, strategy)]
                value = row.get(metric)
                if value is None and metric == "net_pnl_mean":
                    value = float(row["total_return_mean"]) * initial_value
                values.append(scale * float(value))
            axis.bar(x + offset, values, width, label=strategy, color=colors[strategy])
            axis.set_title(title)
            axis.set_ylabel(ylabel)
    for axis, *_ in metrics:
        axis.set_xticks(x, scenarios)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.legend(fontsize=8)

    pnl_width = 0.8 / max(len(all_present), 1)
    for index, strategy in enumerate(all_present):
        offset = (index - (len(all_present) - 1) / 2) * pnl_width
        values = []
        for scenario in scenarios:
            row = lookup[(scenario, strategy)]
            value = row.get("net_pnl_mean")
            if value is None:
                value = float(row["total_return_mean"]) * initial_value
            values.append(float(value))
        axes[1, 1].bar(x + offset, values, pnl_width, label=strategy)
    axes[1, 1].set_xticks(x, scenarios)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Mean cumulative net PnL — all baselines and PPO")
    axes[1, 1].set_ylabel("Net PnL, currency units")
    axes[1, 1].legend(fontsize=7, ncols=2)

    ppo_rows = [lookup[(scenario, "ppo")] for scenario in scenarios]
    axes[2, 0].bar(x - 0.3, [100 * float(row["average_stock_weight_mean"]) for row in ppo_rows], 0.2, label="stock weight, %")
    axes[2, 0].bar(x - 0.1, [100 * float(row["average_abs_futures_exposure_mean"]) for row in ppo_rows], 0.2, label="abs futures, %")
    if "average_option_coverage_mean" in ppo_rows[0]:
        axes[2, 0].bar(
            x + 0.1,
            [100 * float(row["average_option_coverage_mean"]) for row in ppo_rows],
            0.2,
            label="put coverage, %",
        )
    axes[2, 0].bar(x + 0.3, [float(row["turnover_mean"]) for row in ppo_rows], 0.2, label="turnover")
    axes[2, 0].set_xticks(x, scenarios)
    axes[2, 0].set_title("PPO allocation and turnover")
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].axis("off")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_versioned_pnl_comparison(
    summary: Iterable[Mapping[str, Any]],
    registry: Iterable[Mapping[str, Any]],
    *,
    initial_value: float = 1_000_000.0,
    evaluation_split: str = "validation",
) -> Figure:
    """Compare baseline PnL with every registered PPO experiment version."""
    current_rows = list(summary)
    registry_rows = [
        row
        for row in registry
        if str(row.get("evaluation_split", "validation")) == evaluation_split
    ]
    scenarios = [
        name
        for name in ("base", "calm", "crisis_stress")
        if any(str(row["scenario"]) == name for row in current_rows)
    ]
    if not scenarios:
        raise ValueError("Versioned PnL comparison requires evaluation scenarios")

    preferred_baselines = (
        "cash",
        "minimum_stock_unhedged",
        "minimum_stock_put_ladder",
        "signed_rolling_hedge",
    )
    current_lookup = {
        (str(row["scenario"]), str(row["strategy"])): row for row in current_rows
    }
    series: dict[str, list[float]] = {}
    for strategy in preferred_baselines:
        if all((scenario, strategy) in current_lookup for scenario in scenarios):
            values: list[float] = []
            for scenario in scenarios:
                row = current_lookup[(scenario, strategy)]
                pnl = row.get("net_pnl_mean")
                values.append(
                    float(pnl)
                    if pnl is not None
                    else float(row["total_return_mean"]) * initial_value
                )
            series[strategy] = values

    tags = sorted({str(row.get("experiment_tag", "unknown")) for row in registry_rows})
    for tag in tags:
        values = []
        complete = True
        for scenario in scenarios:
            matching = [
                row
                for row in registry_rows
                if str(row.get("experiment_tag")) == tag
                and str(row.get("scenario")) == scenario
            ]
            if not matching:
                complete = False
                break
            pnl_values = []
            for row in matching:
                raw_pnl = row.get("net_pnl_mean")
                if raw_pnl in (None, ""):
                    raw_pnl = float(row["total_return_mean"]) * float(
                        row.get("initial_value", initial_value)
                    )
                pnl_values.append(float(raw_pnl))
            values.append(float(np.mean(pnl_values)))
        if complete:
            series[f"PPO {tag}"] = values

    if not any(label.startswith("PPO ") for label in series):
        raise ValueError("PPO registry contains no complete version across scenarios")

    figure = Figure(figsize=(15, 7))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = np.arange(len(scenarios), dtype=np.float64)
    width = 0.84 / len(series)
    for index, (label, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        axis.bar(x + offset, values, width, label=label)
    axis.set_xticks(x, scenarios)
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_title(f"Cumulative net PnL: baselines vs PPO history ({evaluation_split})")
    axis.set_ylabel("Mean terminal net PnL, currency units")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    figure.tight_layout()
    return figure
