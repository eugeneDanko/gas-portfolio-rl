from datetime import date

import pytest

from gas_portfolio_rl.config.models import MarketConfig
from gas_portfolio_rl.simulation import MarketSimulator
from gas_portfolio_rl.simulation.calendar import crossed_monthly_date, model_date
from gas_portfolio_rl.simulation.seasonality import SeasonalityProfile


def test_dated_profile_has_december_peak_april_trough_and_summer_plateau():
    config = MarketConfig(seasonality_randomization=0.0, seasonality_phase_jitter_days=0)
    profile = SeasonalityProfile(config.seasonality_anchors)

    december_peak = profile.value(date(2027, 12, 27))
    april_trough = profile.value(date(2027, 4, 25))
    june = profile.value(date(2027, 6, 15))
    august = profile.value(date(2027, 8, 15))

    assert december_peak > june > april_trough
    assert june == pytest.approx(august)


def test_model_calendar_tracks_year_and_delivery_dates():
    assert model_date(0, 2027, 252) == date(2027, 1, 1)
    assert model_date(252, 2027, 252) == date(2028, 1, 1)
    state = MarketSimulator().reset(seed=9)
    assert state.calendar_date is not None
    assert state.delivery_date is not None
    assert state.delivery_date > state.calendar_date


def test_monthly_payment_is_detected_when_model_calendar_skips_the_exact_day():
    assert crossed_monthly_date(date(2027, 3, 24), date(2027, 3, 26), 25)
    assert not crossed_monthly_date(date(2027, 3, 26), date(2027, 3, 27), 25)


def test_market_exposes_twelve_contract_forward_curve():
    state = MarketSimulator().reset(seed=12)
    assert len(state.forward_curve) == 12
    assert len(state.forward_delivery_dates) == 12
    assert state.futures_price == pytest.approx(state.forward_curve[0])
    assert all(price > 0.0 for price in state.forward_curve)
    assert list(state.forward_delivery_dates) == sorted(state.forward_delivery_dates)
    assert all(value.day == 27 for value in state.forward_delivery_dates)
    december = [
        price
        for price, delivery in zip(state.forward_curve, state.forward_delivery_dates, strict=True)
        if delivery.month == 12
    ]
    april = [
        price
        for price, delivery in zip(state.forward_curve, state.forward_delivery_dates, strict=True)
        if delivery.month == 4
    ]
    assert december and april and december[0] > april[0]
