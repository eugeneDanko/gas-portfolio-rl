from __future__ import annotations

import numpy as np

from gas_portfolio_rl.config.models import EnvConfig, MarketConfig, OptionConfig
from gas_portfolio_rl.environments.history import HistoryBuffer
from gas_portfolio_rl.portfolio.accounting import Portfolio
from gas_portfolio_rl.simulation.calendar import (
    cyclical_date_features,
    model_date,
    next_contract_date,
)
from gas_portfolio_rl.simulation.models import MarketState


OBSERVATION_NAMES = (
    "sin_day_of_year",
    "cos_day_of_year",
    "sin_delivery_day",
    "cos_delivery_day",
    "gas_return",
    "stock_return",
    "futures_return",
    "stock_volatility",
    "futures_volatility",
    "stock_futures_correlation",
    "futures_basis",
    "curve_slope_1_3",
    "curve_slope_3_6",
    "curve_curvature",
    "winter_summer_spread",
    "time_to_maturity",
    "time_to_dividend",
    "stock_weight",
    "futures_exposure",
    "cash_weight",
    "margin_ratio",
    "drawdown",
    "max_drawdown",
)

OBSERVATION_LOW = np.asarray(
    [-1, -1, -1, -1, -5, -5, -5, 0, 0, -1, -5, -5, -5, -5, -5, 0, 0, -10, -10, -10, 0, -10, 0],
    dtype=np.float32,
)
OBSERVATION_HIGH = np.asarray(
    [1, 1, 1, 1, 5, 5, 5, 10, 10, 1, 5, 5, 5, 5, 5, 1, 1, 10, 10, 10, 10, 0, 1],
    dtype=np.float32,
)

OPTION_OBSERVATION_NAMES = tuple(
    f"put_{family}_{feature}"
    for family in ("short", "medium", "long")
    for feature in (
        "price_ratio",
        "moneyness",
        "time_remaining",
        "implied_volatility",
        "delta",
        "theta_ratio",
        "coverage",
        "unrealized_pnl_ratio",
    )
)
OPTION_OBSERVATION_LOW = np.tile(
    np.asarray([0, 0, 0, 0, -1, -10, 0, -10], dtype=np.float32), 3
)
OPTION_OBSERVATION_HIGH = np.tile(
    np.asarray([2, 5, 1, 2, 0, 10, 2, 10], dtype=np.float32), 3
)


def build_observation(
    market: MarketState,
    portfolio: Portfolio,
    history: HistoryBuffer,
    market_config: MarketConfig,
    env_config: EnvConfig,
    option_config: OptionConfig | None = None,
) -> np.ndarray:
    features = history.features()
    stock_weight, futures_exposure, cash_weight = portfolio.weights(market)
    calendar_date = market.calendar_date or model_date(
        market.day, market_config.calendar_start_year, market_config.trading_days
    )
    delivery_date = market.delivery_date or model_date(
        market.maturity_day, market_config.calendar_start_year, market_config.trading_days
    )
    calendar_sin, calendar_cos = cyclical_date_features(calendar_date)
    delivery_sin, delivery_cos = cyclical_date_features(delivery_date)
    slope_1_3, slope_3_6, curvature, winter_summer = _curve_features(market)
    next_dividend_date = next_contract_date(
        calendar_date, market_config.stock_dividend_payment_day
    )
    time_to_dividend = min((next_dividend_date - calendar_date).days / 31.0, 1.0)
    values = [
        calendar_sin,
        calendar_cos,
        delivery_sin,
        delivery_cos,
        features.gas_return,
        features.stock_return,
        features.futures_return,
        features.stock_volatility,
        features.futures_volatility,
        features.stock_futures_correlation,
        float(np.log(market.futures_price / market.gas_spot)),
        slope_1_3,
        slope_3_6,
        curvature,
        winter_summer,
        market.time_to_maturity,
        time_to_dividend,
        stock_weight,
        futures_exposure,
        cash_weight,
        portfolio.state.margin_ratio,
        portfolio.state.drawdown,
        portfolio.state.max_drawdown_depth,
    ]
    low = OBSERVATION_LOW
    high = OBSERVATION_HIGH
    if option_config is not None and option_config.enabled:
        safe_stock = max(market.stock_price, 1e-12)
        safe_equity = max(abs(portfolio.state.equity), 1e-12)
        for position in portfolio.state.option_positions:
            remaining = max(position.expiry_day - market.day, 0)
            unrealized_pnl = (
                position.mark_price - position.average_entry_price
            ) * position.units
            values.extend(
                [
                    position.mark_price / safe_stock,
                    position.strike / safe_stock,
                    remaining / max(position.tenor_days, 1),
                    position.implied_volatility,
                    position.delta,
                    position.theta / safe_stock,
                    position.coverage(portfolio.state.stock_shares),
                    unrealized_pnl / safe_equity,
                ]
            )
        low = np.concatenate((low, OPTION_OBSERVATION_LOW))
        high = np.concatenate((high, OPTION_OBSERVATION_HIGH))
    if env_config.observable_regime:
        values.append(float(market.regime))
        low = np.append(low, np.float32(0.0))
        high = np.append(high, np.float32(1.0))
    return np.clip(np.asarray(values, dtype=np.float32), low, high)


def _curve_features(market: MarketState) -> tuple[float, float, float, float]:
    if len(market.forward_curve) < 6:
        return 0.0, 0.0, 0.0, 0.0
    log_curve = np.log(np.asarray(market.forward_curve, dtype=np.float64))
    slope_1_3 = float(log_curve[2] - log_curve[0])
    slope_3_6 = float(log_curve[5] - log_curve[2])
    curvature = float(log_curve[2] - 2.0 * log_curve[1] + log_curve[0])
    winter = [
        log_curve[index]
        for index, delivery in enumerate(market.forward_delivery_dates)
        if delivery.month in (12, 1, 2)
    ]
    summer = [
        log_curve[index]
        for index, delivery in enumerate(market.forward_delivery_dates)
        if delivery.month in (6, 7, 8)
    ]
    winter_summer = float(np.mean(winter) - np.mean(summer)) if winter and summer else 0.0
    return slope_1_3, slope_3_6, curvature, winter_summer
