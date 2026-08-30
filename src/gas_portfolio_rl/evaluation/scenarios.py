from __future__ import annotations

from dataclasses import replace

from gas_portfolio_rl.config.models import AppConfig


def make_scenarios(config: AppConfig) -> dict[str, AppConfig]:
    """Build reproducible base, calm and crisis-stress market configurations."""
    calm_market = replace(
        config.market,
        transition_matrix=((0.999, 0.001), (0.12, 0.88)),
        normal=replace(
            config.market.normal,
            gas_volatility=config.market.normal.gas_volatility * 0.75,
            stock_volatility=config.market.normal.stock_volatility * 0.80,
        ),
    )
    crisis_market = replace(
        config.market,
        transition_matrix=((0.97, 0.03), (0.04, 0.96)),
        crisis=replace(
            config.market.crisis,
            gas_volatility=config.market.crisis.gas_volatility * 1.25,
            stock_volatility=config.market.crisis.stock_volatility * 1.25,
            gas_jump_probability=min(config.market.crisis.gas_jump_probability * 1.5, 1.0),
            corporate_jump_probability=min(
                config.market.crisis.corporate_jump_probability * 1.5, 1.0
            ),
        ),
    )
    return {
        "base": config,
        "calm": replace(config, market=calm_market),
        "crisis_stress": replace(config, market=crisis_market),
    }
