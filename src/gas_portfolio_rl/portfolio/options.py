from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from gas_portfolio_rl.config.models import OptionConfig


OPTION_FAMILY_NAMES = ("short", "medium", "long")


@dataclass(frozen=True)
class OptionQuote:
    price: float
    delta: float
    theta: float
    implied_volatility: float


@dataclass(frozen=True)
class OptionPosition:
    family: str
    tenor_days: int
    strike: float
    expiry_day: int
    units: float = 0.0
    mark_price: float = 0.0
    average_entry_price: float = 0.0
    implied_volatility: float = 0.0
    delta: float = 0.0
    theta: float = 0.0

    @property
    def market_value(self) -> float:
        return self.units * self.mark_price

    def coverage(self, stock_shares: float) -> float:
        if stock_shares <= 1e-12:
            return 0.0
        return float(max(self.units, 0.0) / stock_shares)


def initialize_option_ladder(
    stock_price: float,
    current_day: int,
    config: OptionConfig,
    *,
    trading_days: int,
    risk_free_rate: float,
    dividend_yield: float,
    realized_volatility: float = 0.0,
) -> tuple[OptionPosition, ...]:
    positions: list[OptionPosition] = []
    for family, tenor in zip(OPTION_FAMILY_NAMES, config.tenors, strict=True):
        position = OptionPosition(
            family=family,
            tenor_days=tenor,
            strike=config.strike_moneyness * stock_price,
            expiry_day=current_day + tenor,
        )
        positions.append(
            mark_option(
                position,
                stock_price,
                current_day,
                config,
                trading_days=trading_days,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                realized_volatility=realized_volatility,
            )
        )
    return tuple(positions)


def mark_option(
    position: OptionPosition,
    stock_price: float,
    current_day: int,
    config: OptionConfig,
    *,
    trading_days: int,
    risk_free_rate: float,
    dividend_yield: float,
    realized_volatility: float,
) -> OptionPosition:
    remaining_days = max(position.expiry_day - current_day, 0)
    iv = implied_volatility_for_tenor(
        remaining_days=max(remaining_days, 1),
        realized_volatility=realized_volatility,
        config=config,
    )
    quote = european_put_quote(
        spot=stock_price,
        strike=position.strike,
        time_to_expiry=remaining_days / trading_days,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        implied_volatility=iv,
    )
    return replace(
        position,
        mark_price=quote.price,
        implied_volatility=quote.implied_volatility,
        delta=quote.delta,
        theta=quote.theta,
    )


def implied_volatility_for_tenor(
    *,
    remaining_days: int,
    realized_volatility: float,
    config: OptionConfig,
) -> float:
    """Observable IV term structure derived from trailing realized volatility.

    Calm markets have mild term contango. When realized volatility rises above
    the initial IV anchor the slope can invert, making near-term protection
    relatively expensive without exposing the hidden simulator regime.
    """
    base = (
        config.initial_implied_volatility
        if realized_volatility <= 1e-12
        else config.iv_realized_vol_multiplier * realized_volatility
    )
    excess_volatility = max(realized_volatility - config.initial_implied_volatility, 0.0)
    slope = (
        config.iv_term_structure_slope
        - config.iv_crisis_inversion_sensitivity * excess_volatility
    )
    tenor_scale = math.sqrt(remaining_days / max(config.tenors[-1], 1))
    return float(np.clip(base + slope * tenor_scale, config.iv_floor, config.iv_cap))


def european_put_quote(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    implied_volatility: float,
) -> OptionQuote:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if time_to_expiry <= 0:
        payoff = max(strike - spot, 0.0)
        delta = -1.0 if spot < strike else 0.0
        return OptionQuote(payoff, delta, 0.0, implied_volatility)

    sigma = max(implied_volatility, 1e-12)
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * sigma * sigma) * time_to_expiry
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    discount_r = math.exp(-risk_free_rate * time_to_expiry)
    discount_q = math.exp(-dividend_yield * time_to_expiry)
    price = (
        strike * discount_r * _normal_cdf(-d2)
        - spot * discount_q * _normal_cdf(-d1)
    )
    delta = discount_q * (_normal_cdf(d1) - 1.0)
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    theta_annual = (
        -(spot * discount_q * density * sigma) / (2.0 * sqrt_t)
        + risk_free_rate * strike * discount_r * _normal_cdf(-d2)
        - dividend_yield * spot * discount_q * _normal_cdf(-d1)
    )
    return OptionQuote(
        price=max(float(price), 0.0),
        delta=float(delta),
        theta=float(theta_annual),
        implied_volatility=float(implied_volatility),
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
