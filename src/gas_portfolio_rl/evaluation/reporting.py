from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from gas_portfolio_rl.evaluation.evaluator import EvaluationResult
from gas_portfolio_rl.visualization.ppo_diagnostics import plot_evaluation_diagnostics


def save_evaluation(result: EvaluationResult, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "episode_metrics.csv", result.records)
    _write_csv(destination / "summary.csv", result.summary)
    _write_csv(destination / "paired_deltas.csv", result.paired_deltas)
    with (destination / "reward_diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(result.reward_diagnostics, stream, indent=2, ensure_ascii=False)
    _plot_distributions(result.records, destination / "baseline_distributions.png")
    if any(str(row["strategy"]) == "ppo" for row in result.summary):
        ppo_figure = plot_evaluation_diagnostics(result.summary)
        ppo_figure.savefig(destination / "ppo_evaluation_diagnostics.png", dpi=150)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _plot_distributions(records: tuple[dict[str, Any], ...], path: Path) -> None:
    available_scenarios = sorted({str(row["scenario"]) for row in records})
    if not available_scenarios:
        return
    plotted_scenario = "base" if "base" in available_scenarios else available_scenarios[0]
    base = [row for row in records if row["scenario"] == plotted_scenario]
    names = sorted({str(row["strategy"]) for row in base})
    values = [
        np.asarray([float(row["total_return"]) for row in base if row["strategy"] == name])
        for name in names
    ]
    figure = Figure(figsize=(13, 4.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    axes[0].boxplot(values, tick_labels=names, showfliers=False)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title(f"Paired terminal returns: {plotted_scenario} scenario")
    axes[0].set_ylabel("Total return")
    axes[0].tick_params(axis="x", rotation=25)
    mean_drawdowns = [
        np.mean([float(row["max_drawdown"]) for row in base if row["strategy"] == name])
        for name in names
    ]
    axes[1].bar(names, mean_drawdowns)
    axes[1].set_title("Mean maximum drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
