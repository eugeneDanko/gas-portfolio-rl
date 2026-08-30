from __future__ import annotations

from datetime import date
import math

import numpy as np

from gas_portfolio_rl.config.models import MarketConfig, RegimeParameters
from gas_portfolio_rl.simulation.equity_model import advance_stock_price
from gas_portfolio_rl.simulation.calendar import (
    add_months,
    crossed_monthly_date,
    model_date,
    model_trading_day,
    next_contract_date,
)
from gas_portfolio_rl.simulation.futures_model import futures_price
from gas_portfolio_rl.simulation.gas_process import advance_gas_factor
from gas_portfolio_rl.simulation.models import MarketState, MarketTransition, Regime
from gas_portfolio_rl.simulation.regimes import next_regime
from gas_portfolio_rl.simulation.seasonality import SeasonalityProfile


class MarketSimulator:
    """Seeded simulator with a continuous, automatically rolled front contract."""

    def __init__(self, config: MarketConfig | None = None) -> None:
        self.config = config or MarketConfig()
        self.rng = np.random.default_rng()
        self.state: MarketState | None = None
        self._seasonality = SeasonalityProfile(self.config.seasonality_anchors)
        self._volatility_scale = 1.0

    def reset(self, seed: int | None = None) -> MarketState:
        self.rng = np.random.default_rng(seed)
        randomization = self.config.seasonality_randomization
        amplitude_scale = self.rng.uniform(1.0 - randomization, 1.0 + randomization)
        jitter = self.config.seasonality_phase_jitter_days
        phase_shift_days = int(self.rng.integers(-jitter, jitter + 1)) if jitter else 0
        self._seasonality = SeasonalityProfile(
            self.config.seasonality_anchors,
            amplitude_scale=amplitude_scale,
            phase_shift_days=phase_shift_days,
        )
        self._volatility_scale = self.rng.uniform(0.90, 1.10)

        day = int(self.rng.integers(0, self.config.trading_days))
        calendar_date = self._date(day)
        regime = Regime.NORMAL
        factor = -self._seasonality.value(calendar_date)
        front_delivery_date = next_contract_date(
            calendar_date, self.config.contract_expiry_day
        )
        curve, delivery_dates, maturity_days = self._forward_curve(
            day, front_delivery_date, factor
        )
        maturity_day = maturity_days[0]
        future = curve[0]
        self.state = MarketState(
            day=day,
            gas_spot=self.config.initial_gas_price,
            futures_price=future,
            stock_price=self.config.initial_stock_price,
            gas_factor=factor,
            regime=regime,
            maturity_day=maturity_day,
            time_to_maturity=max(maturity_day - day, 0) / self.config.trading_days,
            true_gas_volatility=self._parameters(regime).gas_volatility * self._volatility_scale,
            calendar_date=calendar_date,
            delivery_date=delivery_dates[0],
            forward_curve=curve,
            forward_delivery_dates=delivery_dates,
        )
        return self.state

    def step(self) -> MarketTransition:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")
        previous = self.state
        day = previous.day + 1
        regime = next_regime(previous.regime, self.config.transition_matrix, self.rng)
        parameters = self._parameters(regime)
        dt = 1.0 / self.config.trading_days
        factor = advance_gas_factor(
            previous.gas_factor,
            parameters,
            dt,
            self._volatility_scale,
            self.rng,
        )
        spot = self._spot(day, factor)
        gas_log_return = math.log(spot / previous.gas_spot)
        stock, stock_log_return = advance_stock_price(
            previous.stock_price,
            gas_log_return,
            parameters,
            dt,
            self.rng,
        )
        current_calendar_date = self._date(day)
        assert previous.calendar_date is not None
        is_dividend_payment_date = crossed_monthly_date(
            previous.calendar_date,
            current_calendar_date,
            self.config.stock_dividend_payment_day,
        )
        stock_dividend_per_share = 0.0
        if is_dividend_payment_date:
            stock_dividend_per_share = (
                previous.stock_price * self.config.annual_stock_dividend_yield / 12.0
            )
            stock = max(stock - stock_dividend_per_share, 1e-12)
            stock_log_return = math.log(
                (stock + stock_dividend_per_share) / previous.stock_price
            )

        rolled = day >= previous.maturity_day
        assert previous.delivery_date is not None
        settlement = (
            spot
            if rolled
            else self._future(day, previous.maturity_day, previous.delivery_date, factor)
        )
        front_delivery_date = (
            add_months(previous.delivery_date, 1) if rolled else previous.delivery_date
        )
        curve, delivery_dates, maturity_days = self._forward_curve(
            day, front_delivery_date, factor
        )
        maturity_day = maturity_days[0]
        current_future = curve[0]
        futures_log_return = math.log(settlement / previous.futures_price)
        self.state = MarketState(
            day=day,
            gas_spot=spot,
            futures_price=current_future,
            stock_price=stock,
            gas_factor=factor,
            regime=regime,
            maturity_day=maturity_day,
            time_to_maturity=max(maturity_day - day, 0) / self.config.trading_days,
            true_gas_volatility=parameters.gas_volatility * self._volatility_scale,
            calendar_date=current_calendar_date,
            delivery_date=delivery_dates[0],
            forward_curve=curve,
            forward_delivery_dates=delivery_dates,
            stock_dividend_per_share=stock_dividend_per_share,
            is_dividend_payment_date=is_dividend_payment_date,
        )
        return MarketTransition(
            previous=previous,
            current=self.state,
            futures_settlement_price=settlement,
            gas_log_return=gas_log_return,
            futures_log_return=futures_log_return,
            stock_log_return=stock_log_return,
            rolled=rolled,
        )

    def simulate(self, steps: int, seed: int | None = None) -> list[MarketState]:
        states = [self.reset(seed)]
        for _ in range(steps):
            states.append(self.step().current)
        return states

    def _parameters(self, regime: Regime) -> RegimeParameters:
        return self.config.normal if regime is Regime.NORMAL else self.config.crisis

    def _spot(self, day: int, factor: float) -> float:
        seasonal = self._seasonality.value(self._date(day))
        return math.exp(math.log(self.config.initial_gas_price) + seasonal + factor)

    def _future(
        self,
        day: int,
        maturity_day: int,
        delivery_date: date,
        factor: float,
    ) -> float:
        return futures_price(
            day,
            maturity_day,
            delivery_date,
            factor,
            self.config,
            self._seasonality,
        )

    def _forward_curve(
        self,
        day: int,
        front_delivery_date: date,
        factor: float,
    ) -> tuple[tuple[float, ...], tuple[date, ...], tuple[int, ...]]:
        delivery_dates = tuple(
            add_months(front_delivery_date, offset)
            for offset in range(self.config.forward_curve_contracts)
        )
        maturity_days = tuple(
            model_trading_day(
                delivery_date,
                self.config.calendar_start_year,
                self.config.trading_days,
            )
            for delivery_date in delivery_dates
        )
        prices = tuple(
            self._future(day, maturity, delivery_date, factor)
            for maturity, delivery_date in zip(maturity_days, delivery_dates, strict=True)
        )
        return prices, delivery_dates, maturity_days

    def _date(self, day: int) -> date:
        return model_date(day, self.config.calendar_start_year, self.config.trading_days)
