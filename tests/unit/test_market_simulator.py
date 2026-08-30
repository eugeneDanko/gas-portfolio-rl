import math

import pytest

from gas_portfolio_rl.simulation import MarketSimulator


def _signature(states):
    return [
        (state.day, state.gas_spot, state.futures_price, state.stock_price, int(state.regime))
        for state in states
    ]


def test_seed_is_reproducible_and_different_seeds_differ():
    first = _signature(MarketSimulator().simulate(50, seed=17))
    second = _signature(MarketSimulator().simulate(50, seed=17))
    other = _signature(MarketSimulator().simulate(50, seed=18))
    assert first == second
    assert first != other


def test_expiring_contract_settles_to_spot_before_free_roll():
    simulator = MarketSimulator()
    initial = simulator.reset(seed=7)
    transition = None
    for _ in range(initial.maturity_day - initial.day):
        transition = simulator.step()
    assert transition is not None and transition.rolled
    assert transition.futures_settlement_price == pytest.approx(transition.current.gas_spot)
    assert transition.current.maturity_day > transition.previous.maturity_day


def test_stock_dividend_is_paid_monthly_with_ex_dividend_adjustment():
    simulator = MarketSimulator()
    simulator.reset(seed=13)
    payments = []
    for _ in range(80):
        transition = simulator.step()
        if transition.current.is_dividend_payment_date:
            payments.append(transition)

    assert 2 <= len(payments) <= 5
    for transition in payments:
        expected = (
            transition.previous.stock_price
            * simulator.config.annual_stock_dividend_yield
            / 12.0
        )
        assert transition.current.stock_dividend_per_share == pytest.approx(expected)
        total_return = (
            transition.current.stock_price + transition.current.stock_dividend_per_share
        ) / transition.previous.stock_price
        assert transition.stock_log_return == pytest.approx(math.log(total_return))
