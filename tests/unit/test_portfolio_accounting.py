from dataclasses import replace

import pytest

from gas_portfolio_rl.config.models import PortfolioConfig
from gas_portfolio_rl.portfolio import Portfolio
from gas_portfolio_rl.simulation.models import MarketState, MarketTransition, Regime


def _state(stock=100.0, future=10.0, day=0):
    return MarketState(day, 3.0, future, stock, 0.0, Regime.NORMAL, 21, 21 / 252, 0.3)


def _transition(previous, stock, settlement, dividend_per_share=0.0):
    current = replace(
        previous,
        day=previous.day + 1,
        stock_price=stock,
        futures_price=settlement,
        time_to_maturity=previous.time_to_maturity - 1 / 252,
        stock_dividend_per_share=dividend_per_share,
        is_dividend_payment_date=dividend_per_share > 0.0,
    )
    return MarketTransition(previous, current, settlement, 0.0, 0.0, 0.0, False)


def _zero_cost_config(**overrides):
    values = dict(stock_transaction_cost=0.0, futures_transaction_cost=0.0, annual_cash_rate=0.0)
    values.update(overrides)
    return PortfolioConfig(**values)


def test_margin_is_not_subtracted_from_equity():
    portfolio = Portfolio(_zero_cost_config())
    portfolio.rebalance(0.0, 1.0, _state())
    assert portfolio.state.equity == pytest.approx(1_000_000.0)
    assert portfolio.state.cash == pytest.approx(1_000_000.0)
    assert portfolio.state.margin_required == pytest.approx(150_000.0)


@pytest.mark.parametrize(
    ("exposure", "next_future", "expected_pnl"),
    [(1.0, 11.0, 100_000.0), (-1.0, 9.0, 100_000.0)],
)
def test_long_and_short_futures_have_correct_pnl_sign(exposure, next_future, expected_pnl):
    market = _state()
    portfolio = Portfolio(_zero_cost_config())
    portfolio.rebalance(0.0, exposure, market)
    result = portfolio.settle(_transition(market, stock=100.0, settlement=next_future))
    assert result.futures_pnl == pytest.approx(expected_pnl)
    assert portfolio.state.equity == pytest.approx(1_000_000.0 + expected_pnl)


def test_flat_portfolio_only_earns_cash_rate():
    market = _state()
    portfolio = Portfolio(_zero_cost_config(annual_cash_rate=0.0252), trading_days=252)
    portfolio.rebalance(0.0, 0.0, market)
    portfolio.settle(_transition(market, stock=120.0, settlement=15.0))
    assert portfolio.state.equity == pytest.approx(1_000_100.0)


def test_transaction_cost_reduces_equity_once():
    market = _state()
    portfolio = Portfolio(PortfolioConfig(annual_cash_rate=0.0))
    result = portfolio.rebalance(0.5, 0.5, market)
    assert result.transaction_cost == pytest.approx(350.0)
    assert portfolio.state.equity == pytest.approx(999_650.0)


def test_stock_dividend_is_paid_to_cash_and_equity():
    market = _state()
    portfolio = Portfolio(_zero_cost_config(), trading_days=252)
    portfolio.rebalance(1.0, 0.0, market)

    result = portfolio.settle(
        _transition(market, stock=99.75, settlement=10.0, dividend_per_share=0.25)
    )

    assert result.stock_dividend == pytest.approx(2_500.0)
    assert portfolio.state.cash == pytest.approx(2_500.0)
    assert portfolio.state.equity == pytest.approx(1_000_000.0)
    assert portfolio.state.cumulative_dividends == pytest.approx(2_500.0)


def test_stock_dividend_is_zero_between_monthly_payments():
    market = _state()
    portfolio = Portfolio(_zero_cost_config())
    portfolio.rebalance(1.0, 0.0, market)

    result = portfolio.settle(_transition(market, stock=100.0, settlement=10.0))

    assert result.stock_dividend == 0.0
    assert portfolio.state.cumulative_dividends == 0.0


def test_margin_call_fully_liquidates_futures_position():
    market = _state()
    portfolio = Portfolio(_zero_cost_config())
    portfolio.rebalance(0.8, 1.0, market)

    result = portfolio.settle(_transition(market, stock=100.0, settlement=9.0))

    assert result.margin_call
    assert portfolio.state.futures_contracts == pytest.approx(0.0)
    assert portfolio.state.futures_target_notional == pytest.approx(0.0)
    assert portfolio.state.margin_required == pytest.approx(0.0)
