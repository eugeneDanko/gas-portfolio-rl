from __future__ import annotations

import numpy as np

from gas_portfolio_rl.config.models import RegimeParameters


def advance_gas_factor(
    factor: float,
    parameters: RegimeParameters,
    dt: float,
    volatility_scale: float,
    rng: np.random.Generator,
) -> float:
    diffusion = parameters.gas_volatility * volatility_scale * np.sqrt(dt) * rng.normal()
    jump = 0.0
    if rng.random() < parameters.gas_jump_probability:
        jump = rng.normal(parameters.gas_jump_mean, parameters.gas_jump_std)
    return float(
        factor
        + parameters.kappa * (parameters.long_run_factor - factor) * dt
        + diffusion
        + jump
    )

