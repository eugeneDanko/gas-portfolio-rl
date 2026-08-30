from __future__ import annotations

import math

import numpy as np

from gas_portfolio_rl.config.models import RegimeParameters


def advance_stock_price(
    stock_price: float,
    gas_log_return: float,
    parameters: RegimeParameters,
    dt: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    market_return = (
        parameters.market_drift * dt
        + parameters.market_volatility * math.sqrt(dt) * rng.normal()
    )
    corporate_jump = 0.0
    if rng.random() < parameters.corporate_jump_probability:
        corporate_jump = rng.normal(
            parameters.corporate_jump_mean,
            parameters.corporate_jump_std,
        )
    stock_log_return = (
        parameters.stock_drift * dt
        + parameters.gas_beta * gas_log_return
        + market_return
        + parameters.stock_volatility * math.sqrt(dt) * rng.normal()
        + corporate_jump
    )
    return stock_price * math.exp(stock_log_return), float(stock_log_return)

