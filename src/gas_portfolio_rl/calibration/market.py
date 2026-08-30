from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from gas_portfolio_rl.config.models import CalibrationConfig, MarketConfig
from gas_portfolio_rl.simulation.market_simulator import MarketSimulator
from gas_portfolio_rl.simulation.models import Regime


@dataclass(frozen=True)
class MarketMoments:
    gas_volatility: float
    stock_volatility: float
    futures_volatility: float
    stock_futures_correlation: float
    crisis_fraction: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationErrors:
    gas_volatility_relative: float
    stock_volatility_relative: float
    futures_volatility_relative: float
    stock_futures_correlation_absolute: float
    crisis_fraction_absolute: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationResult:
    market_config: MarketConfig
    moments: MarketMoments
    objective: float
    candidates_evaluated: int
    maximum_objective: float
    errors: CalibrationErrors
    individual_moments_pass: bool
    targets: MarketMoments
    tolerances: CalibrationErrors

    @property
    def passes_gate(self) -> bool:
        return self.objective <= self.maximum_objective and self.individual_moments_pass


def estimate_market_moments(
    market_config: MarketConfig,
    *,
    seeds: list[int] | tuple[int, ...],
    steps: int,
) -> MarketMoments:
    gas_returns: list[float] = []
    stock_returns: list[float] = []
    futures_returns: list[float] = []
    regimes: list[float] = []
    simulator = MarketSimulator(market_config)
    for seed in seeds:
        simulator.reset(seed)
        for _ in range(steps):
            transition = simulator.step()
            gas_returns.append(transition.gas_log_return)
            stock_returns.append(transition.stock_log_return)
            futures_returns.append(transition.futures_log_return)
            regimes.append(float(transition.current.regime is Regime.CRISIS))

    scale = math.sqrt(market_config.trading_days)
    gas = np.asarray(gas_returns)
    stock = np.asarray(stock_returns)
    future = np.asarray(futures_returns)
    correlation = 0.0
    if np.std(stock) > 1e-12 and np.std(future) > 1e-12:
        correlation = float(np.corrcoef(stock, future)[0, 1])
    return MarketMoments(
        gas_volatility=float(np.std(gas, ddof=1) * scale),
        stock_volatility=float(np.std(stock, ddof=1) * scale),
        futures_volatility=float(np.std(future, ddof=1) * scale),
        stock_futures_correlation=correlation,
        crisis_fraction=float(np.mean(regimes)),
    )


def calibrate_market(
    market_config: MarketConfig,
    calibration: CalibrationConfig,
) -> CalibrationResult:
    """Seeded simulation-based calibration over a compact parameter search."""
    rng = np.random.default_rng(calibration.seed)
    path_seeds = [int(value) for value in rng.integers(0, 2**31 - 1,
                                                       calibration.paths_per_candidate)]
    candidates = [market_config]
    for _ in range(calibration.candidate_count - 1):
        normal = replace(
            market_config.normal,
            gas_volatility=float(rng.uniform(0.25, 0.60)),
            stock_volatility=float(rng.uniform(0.05, 0.24)),
            gas_beta=float(rng.uniform(0.05, 0.75)),
            market_volatility=float(rng.uniform(0.07, 0.14)),
        )
        crisis = replace(
            market_config.crisis,
            gas_volatility=float(rng.uniform(0.60, 1.00)),
            stock_volatility=float(rng.uniform(0.15, 0.42)),
            market_volatility=float(rng.uniform(0.15, 0.35)),
            corporate_jump_probability=float(rng.uniform(0.003, 0.015)),
        )
        normal_to_crisis = float(rng.uniform(0.002, 0.015))
        crisis_to_normal = float(rng.uniform(0.02, 0.09))
        candidates.append(
            replace(
                market_config,
                normal=normal,
                crisis=crisis,
                transition_matrix=(
                    (1.0 - normal_to_crisis, normal_to_crisis),
                    (crisis_to_normal, 1.0 - crisis_to_normal),
                ),
            )
        )

    best: CalibrationResult | None = None
    for candidate in candidates:
        moments = estimate_market_moments(
            candidate, seeds=path_seeds, steps=calibration.path_steps
        )
        objective = _calibration_objective(moments, calibration)
        errors = _calibration_errors(moments, calibration)
        individual_pass = _individual_moments_pass(errors, calibration)
        targets = _target_moments(calibration)
        tolerances = _calibration_tolerances(calibration)
        candidate_result = CalibrationResult(
            candidate,
            moments,
            objective,
            len(candidates),
            calibration.maximum_objective,
            errors,
            individual_pass,
            targets,
            tolerances,
        )
        if best is None or (
            not candidate_result.individual_moments_pass,
            candidate_result.objective,
        ) < (not best.individual_moments_pass, best.objective):
            best = candidate_result
    assert best is not None
    return best


def _calibration_objective(moments: MarketMoments, targets: CalibrationConfig) -> float:
    errors = (
        (moments.gas_volatility - targets.gas_volatility_target)
        / targets.gas_volatility_target,
        (moments.stock_volatility - targets.stock_volatility_target)
        / targets.stock_volatility_target,
        (moments.futures_volatility - targets.futures_volatility_target)
        / targets.futures_volatility_target,
        (moments.stock_futures_correlation - targets.stock_futures_correlation_target) / 0.25,
        (moments.crisis_fraction - targets.crisis_fraction_target) / 0.10,
    )
    return float(np.mean(np.square(errors)))


def _calibration_errors(
    moments: MarketMoments, targets: CalibrationConfig
) -> CalibrationErrors:
    return CalibrationErrors(
        gas_volatility_relative=abs(
            moments.gas_volatility - targets.gas_volatility_target
        ) / targets.gas_volatility_target,
        stock_volatility_relative=abs(
            moments.stock_volatility - targets.stock_volatility_target
        ) / targets.stock_volatility_target,
        futures_volatility_relative=abs(
            moments.futures_volatility - targets.futures_volatility_target
        ) / targets.futures_volatility_target,
        stock_futures_correlation_absolute=abs(
            moments.stock_futures_correlation
            - targets.stock_futures_correlation_target
        ),
        crisis_fraction_absolute=abs(
            moments.crisis_fraction - targets.crisis_fraction_target
        ),
    )


def _individual_moments_pass(
    errors: CalibrationErrors, config: CalibrationConfig
) -> bool:
    return bool(
        errors.gas_volatility_relative <= config.gas_volatility_relative_tolerance
        and errors.stock_volatility_relative <= config.stock_volatility_relative_tolerance
        and errors.futures_volatility_relative
        <= config.futures_volatility_relative_tolerance
        and errors.stock_futures_correlation_absolute
        <= config.stock_futures_correlation_absolute_tolerance
        and errors.crisis_fraction_absolute <= config.crisis_fraction_absolute_tolerance
    )


def _target_moments(config: CalibrationConfig) -> MarketMoments:
    return MarketMoments(
        gas_volatility=config.gas_volatility_target,
        stock_volatility=config.stock_volatility_target,
        futures_volatility=config.futures_volatility_target,
        stock_futures_correlation=config.stock_futures_correlation_target,
        crisis_fraction=config.crisis_fraction_target,
    )


def _calibration_tolerances(config: CalibrationConfig) -> CalibrationErrors:
    return CalibrationErrors(
        gas_volatility_relative=config.gas_volatility_relative_tolerance,
        stock_volatility_relative=config.stock_volatility_relative_tolerance,
        futures_volatility_relative=config.futures_volatility_relative_tolerance,
        stock_futures_correlation_absolute=(
            config.stock_futures_correlation_absolute_tolerance
        ),
        crisis_fraction_absolute=config.crisis_fraction_absolute_tolerance,
    )
