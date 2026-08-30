from __future__ import annotations

import math
from datetime import date

from gas_portfolio_rl.config.models import MarketConfig
from gas_portfolio_rl.simulation.seasonality import SeasonalityProfile


def futures_price(
    day: int,
    maturity_day: int,
    delivery_date: date,
    gas_factor: float,
    config: MarketConfig,
    seasonality: SeasonalityProfile,
) -> float:
    tau = max(maturity_day - day, 0) / config.trading_days
    decay = math.exp(-config.q_kappa * tau)
    variance = (config.q_sigma**2 / (4.0 * config.q_kappa)) * (
        1.0 - math.exp(-2.0 * config.q_kappa * tau)
    )
    seasonal_at_maturity = seasonality.value(delivery_date)
    log_price = (
        math.log(config.initial_gas_price)
        + seasonal_at_maturity
        + gas_factor * decay
        + config.q_theta * (1.0 - decay)
        + variance
    )
    return math.exp(log_price)
