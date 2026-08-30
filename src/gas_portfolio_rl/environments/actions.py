from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gas_portfolio_rl.config.models import OptionConfig, PortfolioConfig


@dataclass(frozen=True)
class ProjectedAction:
    stock_weight: float
    futures_exposure: float
    option_coverages: tuple[float, float, float]
    was_projected: bool


def project_action(
    action: np.ndarray,
    config: PortfolioConfig,
    option_config: OptionConfig | None = None,
) -> ProjectedAction:
    raw = np.asarray(action, dtype=np.float64).reshape(-1)
    options = option_config or OptionConfig()
    valid_sizes = {2, 5} if options.enabled else {2, 5}
    if raw.size not in valid_sizes or not np.all(np.isfinite(raw)):
        raise ValueError("Action must contain two or five finite values")
    stock_weight = float(np.clip(raw[0], 0.0, 1.0))
    futures_exposure = float(np.clip(raw[1], -1.0, 1.0))
    coverages = np.zeros(3, dtype=np.float64)
    if options.enabled and raw.size == 5:
        coverages = np.clip(raw[2:5], 0.0, options.maximum_total_coverage)
        total_coverage = float(np.sum(coverages))
        if total_coverage > options.maximum_total_coverage:
            coverages *= options.maximum_total_coverage / total_coverage
    max_usage = 1.0 - config.min_cash_buffer
    usage = stock_weight + config.margin_rate * abs(futures_exposure)
    if usage > max_usage:
        scale = max_usage / usage
        stock_weight *= scale
        futures_exposure *= scale
    projected_values = np.concatenate(([stock_weight, futures_exposure], coverages))
    comparable_raw = raw if raw.size == 5 else np.concatenate((raw, np.zeros(3)))
    projected = not np.allclose(
        comparable_raw, projected_values, rtol=0.0, atol=1e-8
    )
    return ProjectedAction(
        stock_weight,
        futures_exposure,
        tuple(float(value) for value in coverages),
        projected,
    )
