from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class RegimeParameters:
    kappa: float
    long_run_factor: float
    gas_volatility: float
    gas_jump_probability: float
    gas_jump_mean: float
    gas_jump_std: float
    stock_drift: float
    stock_volatility: float
    gas_beta: float
    market_drift: float
    market_volatility: float
    corporate_jump_probability: float
    corporate_jump_mean: float
    corporate_jump_std: float


def _normal_regime() -> RegimeParameters:
    return RegimeParameters(2.5, 0.0, 0.35, 0.001, 0.0, 0.12, 0.07, 0.18, 0.35, 0.05, 0.12, 0.0005, -0.08, 0.05)


def _crisis_regime() -> RegimeParameters:
    return RegimeParameters(1.2, 0.20, 0.75, 0.020, 0.04, 0.20, -0.18, 0.42, 0.20, -0.25, 0.35, 0.015, -0.12, 0.10)


def _seasonality_anchors() -> tuple[tuple[str, float], ...]:
    """Interpretable log-seasonality profile by delivery month and day."""
    return (
        ("01-01", 0.30),
        ("02-15", 0.10),
        ("04-25", -0.22),
        ("06-01", 0.02),
        ("08-31", 0.02),
        ("10-15", 0.10),
        ("12-15", 0.24),
        ("12-27", 0.42),
        ("12-31", 0.34),
    )


@dataclass(frozen=True)
class MarketConfig:
    trading_days: int = 252
    episode_steps: int = 504
    calendar_start_year: int = 2027
    initial_gas_price: float = 3.0
    initial_stock_price: float = 100.0
    annual_stock_dividend_yield: float = 0.03
    stock_dividend_payment_day: int = 25
    seasonality_anchors: tuple[tuple[str, float], ...] = field(default_factory=_seasonality_anchors)
    seasonality_randomization: float = 0.10
    seasonality_phase_jitter_days: int = 3
    contract_expiry_day: int = 27
    forward_curve_contracts: int = 12
    q_kappa: float = 2.0
    q_theta: float = 0.02
    q_sigma: float = 0.40
    transition_matrix: tuple[tuple[float, float], tuple[float, float]] = (
        (0.995, 0.005),
        (0.030, 0.970),
    )
    normal: RegimeParameters = field(default_factory=_normal_regime)
    crisis: RegimeParameters = field(default_factory=_crisis_regime)

    def __post_init__(self) -> None:
        if self.trading_days <= 0 or self.episode_steps <= 0:
            raise ValueError("Time dimensions must be positive")
        if not 1 <= self.contract_expiry_day <= 28:
            raise ValueError("contract_expiry_day must be in [1, 28]")
        if self.initial_gas_price <= 0 or self.initial_stock_price <= 0:
            raise ValueError("Initial prices must be positive")
        if self.annual_stock_dividend_yield < 0:
            raise ValueError("annual_stock_dividend_yield must be non-negative")
        if not 1 <= self.stock_dividend_payment_day <= 28:
            raise ValueError("stock_dividend_payment_day must be in [1, 28]")
        if self.calendar_start_year < 1900:
            raise ValueError("calendar_start_year must be at least 1900")
        if not 0.0 <= self.seasonality_randomization < 1.0:
            raise ValueError("seasonality_randomization must be in [0, 1)")
        if self.seasonality_phase_jitter_days < 0:
            raise ValueError("seasonality_phase_jitter_days must be non-negative")
        if self.forward_curve_contracts < 12:
            raise ValueError("forward_curve_contracts must be at least 12")
        if len(self.seasonality_anchors) < 3:
            raise ValueError("seasonality_anchors must contain at least three points")
        anchor_dates: set[str] = set()
        for month_day, _ in self.seasonality_anchors:
            datetime.strptime(f"2001-{month_day}", "%Y-%m-%d")
            if month_day in anchor_dates:
                raise ValueError(f"Duplicate seasonality anchor: {month_day}")
            anchor_dates.add(month_day)
        if len(self.transition_matrix) != 2 or any(len(row) != 2 for row in self.transition_matrix):
            raise ValueError("transition_matrix must be 2x2")
        for row in self.transition_matrix:
            if any(value < 0 for value in row) or abs(sum(row) - 1.0) > 1e-9:
                raise ValueError("Every transition row must be a probability vector")


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 1_000_000.0
    margin_rate: float = 0.15
    stock_transaction_cost: float = 0.0005
    futures_transaction_cost: float = 0.0002
    annual_cash_rate: float = 0.02
    min_cash_buffer: float = 0.01
    bankruptcy_fraction: float = 0.20

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 < self.margin_rate <= 1:
            raise ValueError("margin_rate must be in (0, 1]")
        if not 0 <= self.min_cash_buffer < 1:
            raise ValueError("min_cash_buffer must be in [0, 1)")


@dataclass(frozen=True)
class OptionConfig:
    """Tradable long-put ladder used to insure the GasCo stock position."""

    enabled: bool = False
    tenors: tuple[int, int, int] = (10, 21, 63)
    strike_moneyness: float = 0.95
    maximum_total_coverage: float = 1.0
    maximum_option_value_fraction: float = 0.10
    maximum_annual_premium_fraction: float = 0.08
    contract_multiplier: float = 100.0
    commission_per_contract: float = 1.0
    relative_half_spread: float = 0.01
    initial_implied_volatility: float = 0.30
    iv_realized_vol_multiplier: float = 1.20
    iv_floor: float = 0.15
    iv_cap: float = 1.00
    iv_term_structure_slope: float = 0.04
    iv_crisis_inversion_sensitivity: float = 0.50

    def __post_init__(self) -> None:
        if len(self.tenors) != 3 or tuple(sorted(self.tenors)) != self.tenors:
            raise ValueError("option tenors must contain three increasing maturities")
        if any(value < 2 for value in self.tenors):
            raise ValueError("option tenors must be at least two trading days")
        if not 0 < self.strike_moneyness <= 2:
            raise ValueError("strike_moneyness must be in (0, 2]")
        if not 0 <= self.maximum_total_coverage <= 1:
            raise ValueError("maximum_total_coverage must be in [0, 1]")
        if not 0 < self.maximum_option_value_fraction < 1:
            raise ValueError("maximum_option_value_fraction must be in (0, 1)")
        if not 0 < self.maximum_annual_premium_fraction < 1:
            raise ValueError("maximum_annual_premium_fraction must be in (0, 1)")
        if self.contract_multiplier <= 0 or self.commission_per_contract < 0:
            raise ValueError("option contract multiplier must be positive and commission non-negative")
        if not 0 <= self.relative_half_spread < 1:
            raise ValueError("relative_half_spread must be in [0, 1)")
        if not 0 < self.iv_floor <= self.initial_implied_volatility <= self.iv_cap:
            raise ValueError("option IV values must satisfy 0 < floor <= initial <= cap")
        if self.iv_realized_vol_multiplier <= 0:
            raise ValueError("iv_realized_vol_multiplier must be positive")
        if self.iv_crisis_inversion_sensitivity < 0:
            raise ValueError("iv_crisis_inversion_sensitivity must be non-negative")


@dataclass(frozen=True)
class EnvConfig:
    rolling_window: int = 20
    observable_regime: bool = False

    def __post_init__(self) -> None:
        if self.rolling_window < 2:
            raise ValueError("rolling_window must be at least 2")


@dataclass(frozen=True)
class RewardConfig:
    profit_weight: float = 1.0
    loss_aversion: float = 1.25
    hedge_effectiveness_weight: float = 0.75
    hedge_effectiveness_clip: float = 0.02
    downside_penalty: float = 2.0
    drawdown_increase_penalty: float = 0.10
    drawdown_budget_penalty: float = 0.50
    turnover_penalty: float = 0.002
    safe_margin_utilization: float = 0.08
    margin_utilization_penalty: float = 2.0
    maximum_drawdown: float = 0.15
    terminal_shortfall_penalty: float = 1.0
    terminal_drawdown_penalty: float = 2.0
    margin_call_penalty: float = 5.0
    margin_call_loss_penalty: float = 2.0
    early_termination_penalty: float = 4.0
    bankruptcy_penalty: float = 8.0
    bankruptcy_loss_penalty: float = 2.0

    def __post_init__(self) -> None:
        if self.loss_aversion < 1:
            raise ValueError("loss_aversion must be at least 1")
        if self.hedge_effectiveness_clip <= 0:
            raise ValueError("hedge_effectiveness_clip must be positive")
        if not 0 <= self.safe_margin_utilization <= 1:
            raise ValueError("safe_margin_utilization must be in [0, 1]")
        if not 0 <= self.maximum_drawdown <= 1:
            raise ValueError("maximum_drawdown must be in [0, 1]")
        penalties = (
            self.profit_weight,
            self.hedge_effectiveness_weight,
            self.downside_penalty,
            self.drawdown_increase_penalty,
            self.drawdown_budget_penalty,
            self.turnover_penalty,
            self.margin_utilization_penalty,
            self.terminal_shortfall_penalty,
            self.terminal_drawdown_penalty,
            self.margin_call_penalty,
            self.margin_call_loss_penalty,
            self.early_termination_penalty,
            self.bankruptcy_penalty,
            self.bankruptcy_loss_penalty,
        )
        if any(value < 0 for value in penalties):
            raise ValueError("Reward penalties must be non-negative")


@dataclass(frozen=True)
class RelativeMomentumConfig:
    lookback: int = 20
    allocation: float = 0.50
    neutral_stock_weight: float = 0.25
    score_threshold: float = 0.25
    rebalance_interval: int = 5

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0 <= self.allocation <= 1:
            raise ValueError("allocation must be in [0, 1]")
        if not 0 <= self.neutral_stock_weight <= 1:
            raise ValueError("neutral_stock_weight must be in [0, 1]")
        if self.score_threshold < 0 or self.rebalance_interval < 1:
            raise ValueError("score_threshold must be non-negative and interval positive")


@dataclass(frozen=True)
class RollingHedgeConfig:
    lookback: int = 20
    stock_weight: float = 0.50
    initial_hedge_ratio: float = 0.50
    maximum_futures_exposure: float = 0.50
    rebalance_interval: int = 5

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0 <= self.stock_weight <= 1:
            raise ValueError("stock_weight must be in [0, 1]")
        if self.initial_hedge_ratio < 0:
            raise ValueError("initial_hedge_ratio must be non-negative")
        if not 0 <= self.maximum_futures_exposure <= 1:
            raise ValueError("maximum_futures_exposure must be in [0, 1]")
        if self.rebalance_interval < 1:
            raise ValueError("rebalance_interval must be positive")


@dataclass(frozen=True)
class SignedRollingHedgeConfig:
    lookback: int = 40
    stock_weight: float = 0.50
    initial_hedge_ratio: float = 0.50
    maximum_futures_exposure: float = 0.50
    rebalance_interval: int = 5
    beta_smoothing: float = 0.20
    beta_shrinkage: float = 0.50
    minimum_abs_correlation: float = 0.10
    exposure_change_threshold: float = 0.03

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0 <= self.stock_weight <= 1:
            raise ValueError("stock_weight must be in [0, 1]")
        if self.initial_hedge_ratio < 0:
            raise ValueError("initial_hedge_ratio must be non-negative")
        if not 0 <= self.maximum_futures_exposure <= 1:
            raise ValueError("maximum_futures_exposure must be in [0, 1]")
        if self.rebalance_interval < 1:
            raise ValueError("rebalance_interval must be positive")
        if not 0 < self.beta_smoothing <= 1:
            raise ValueError("beta_smoothing must be in (0, 1]")
        if not 0 <= self.beta_shrinkage <= 1:
            raise ValueError("beta_shrinkage must be in [0, 1]")
        if not 0 <= self.minimum_abs_correlation <= 1:
            raise ValueError("minimum_abs_correlation must be in [0, 1]")
        if self.exposure_change_threshold < 0:
            raise ValueError("exposure_change_threshold must be non-negative")


@dataclass(frozen=True)
class BaselinesConfig:
    relative_momentum: RelativeMomentumConfig = field(default_factory=RelativeMomentumConfig)
    rolling_hedge: RollingHedgeConfig = field(default_factory=RollingHedgeConfig)
    signed_rolling_hedge: SignedRollingHedgeConfig = field(
        default_factory=SignedRollingHedgeConfig
    )


@dataclass(frozen=True)
class CalibrationConfig:
    seed: int = 2027
    paths_per_candidate: int = 24
    candidate_count: int = 32
    path_steps: int = 504
    gas_volatility_target: float = 0.45
    stock_volatility_target: float = 0.24
    futures_volatility_target: float = 0.42
    stock_futures_correlation_target: float = 0.20
    crisis_fraction_target: float = 0.14
    maximum_objective: float = 0.10
    gas_volatility_relative_tolerance: float = 0.15
    stock_volatility_relative_tolerance: float = 0.15
    futures_volatility_relative_tolerance: float = 0.15
    stock_futures_correlation_absolute_tolerance: float = 0.08
    crisis_fraction_absolute_tolerance: float = 0.03

    def __post_init__(self) -> None:
        if self.paths_per_candidate < 1 or self.candidate_count < 1 or self.path_steps < 20:
            raise ValueError("Calibration sizes must be positive and path_steps at least 20")
        if min(self.gas_volatility_target, self.stock_volatility_target,
               self.futures_volatility_target) <= 0:
            raise ValueError("Volatility targets must be positive")
        if not -1 <= self.stock_futures_correlation_target <= 1:
            raise ValueError("Correlation target must be in [-1, 1]")
        if not 0 <= self.crisis_fraction_target <= 1:
            raise ValueError("crisis_fraction_target must be in [0, 1]")
        if self.maximum_objective <= 0:
            raise ValueError("maximum_objective must be positive")
        tolerances = (
            self.gas_volatility_relative_tolerance,
            self.stock_volatility_relative_tolerance,
            self.futures_volatility_relative_tolerance,
            self.stock_futures_correlation_absolute_tolerance,
            self.crisis_fraction_absolute_tolerance,
        )
        if any(value <= 0 for value in tolerances):
            raise ValueError("Calibration tolerances must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    seed_start: int = 10_000
    seeds: int = 200
    stress_seeds: int = 100
    confidence_level: float = 0.95
    cvar_alpha: float = 0.05
    bootstrap_samples: int = 500
    fixed_stock_weight: float = 0.50
    fixed_futures_exposure: float = -0.25
    reward_grid_seeds: int = 20

    def __post_init__(self) -> None:
        if (
            self.seeds < 1
            or self.stress_seeds < 1
            or self.reward_grid_seeds < 1
            or self.bootstrap_samples < 1
        ):
            raise ValueError("Evaluation sample sizes must be positive")
        if not 0 < self.confidence_level < 1 or not 0 < self.cvar_alpha < 1:
            raise ValueError("confidence_level and cvar_alpha must be in (0, 1)")
        if not 0 <= self.fixed_stock_weight <= 1:
            raise ValueError("fixed_stock_weight must be in [0, 1]")
        if not -1 <= self.fixed_futures_exposure <= 1:
            raise ValueError("fixed_futures_exposure must be in [-1, 1]")


@dataclass(frozen=True)
class PPOConfig:
    version: str = "v3"
    experiment_tag: str = "v5_full"
    total_timesteps: int = 600_000
    n_envs: int = 4
    learning_rate: float = 2e-4
    learning_rate_schedule: str = "linear"
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_range: float = 0.15
    ent_coef: float = 0.002
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50
    policy_layers: tuple[int, ...] = (128, 128)
    normalize_observations: bool = True
    normalize_rewards: bool = True
    clip_observation: float = 10.0
    clip_reward: float = 10.0
    eval_frequency: int = 20_000
    validation_episodes: int = 32
    validation_stress_episodes: int = 32
    crisis_validation_weight: float = 0.40
    checkpoint_frequency: int = 50_000
    decision_interval: int = 5
    stock_action_deadband: float = 0.03
    futures_action_deadband: float = 0.03
    option_coverage_deadband: float = 0.06
    minimum_option_coverage: float = 0.25
    option_target_smoothing: float = 0.35
    minimum_stock_weight: float = 0.25
    maximum_stock_weight: float = 0.75
    maximum_futures_exposure: float = 0.50
    stress_training_fraction: float = 0.50
    margin_call_score_penalty: float = 5.0
    bankruptcy_score_penalty: float = 10.0
    maximum_high_cash_fraction: float = 0.60
    high_cash_score_penalty: float = 1.0
    minimum_average_stock_weight: float = 0.10
    stock_shortfall_score_penalty: float = 2.0
    maximum_turnover: float = 20.0
    turnover_score_penalty: float = 0.05
    put_ladder_drawdown_score_penalty: float = 4.0
    device: str = "auto"

    def __post_init__(self) -> None:
        positive_ints = (
            self.total_timesteps,
            self.n_envs,
            self.n_steps,
            self.batch_size,
            self.n_epochs,
            self.eval_frequency,
            self.validation_episodes,
            self.validation_stress_episodes,
            self.checkpoint_frequency,
            self.decision_interval,
        )
        if any(value < 1 for value in positive_ints):
            raise ValueError("PPO sizes and frequencies must be positive")
        if self.batch_size > self.n_steps * self.n_envs:
            raise ValueError("batch_size cannot exceed n_steps * n_envs")
        if (self.n_steps * self.n_envs) % self.batch_size != 0:
            raise ValueError("batch_size must divide n_steps * n_envs")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.version not in {"v1", "v2", "v3", "v4"}:
            raise ValueError("PPO version must be 'v1', 'v2', 'v3' or 'v4'")
        if not self.experiment_tag.strip():
            raise ValueError("experiment_tag must be non-empty")
        if self.learning_rate_schedule not in {"constant", "linear"}:
            raise ValueError("learning_rate_schedule must be constant or linear")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        if self.clip_range <= 0 or self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError("Invalid PPO loss coefficients")
        if not self.policy_layers or any(width < 1 for width in self.policy_layers):
            raise ValueError("policy_layers must contain positive widths")
        if self.clip_observation <= 0 or self.clip_reward <= 0:
            raise ValueError("Normalization clips must be positive")
        if not 0 <= self.crisis_validation_weight <= 1:
            raise ValueError("crisis_validation_weight must be in [0, 1]")
        if not 0 <= self.minimum_stock_weight <= self.maximum_stock_weight <= 1:
            raise ValueError("PPO stock limits must satisfy 0 <= minimum <= maximum <= 1")
        if not 0 <= self.maximum_futures_exposure <= 1:
            raise ValueError("maximum_futures_exposure must be in [0, 1]")
        if not 0 <= self.minimum_option_coverage <= 1:
            raise ValueError("minimum_option_coverage must be in [0, 1]")
        if not 0 < self.option_target_smoothing <= 1:
            raise ValueError("option_target_smoothing must be in (0, 1]")
        if not 0 <= self.stress_training_fraction <= 1:
            raise ValueError("stress_training_fraction must be in [0, 1]")
        if self.margin_call_score_penalty < 0 or self.bankruptcy_score_penalty < 0:
            raise ValueError("Validation score penalties must be non-negative")
        if not 0 <= self.maximum_high_cash_fraction <= 1:
            raise ValueError("maximum_high_cash_fraction must be in [0, 1]")
        if self.high_cash_score_penalty < 0:
            raise ValueError("high_cash_score_penalty must be non-negative")
        if not 0 <= self.minimum_average_stock_weight <= 1:
            raise ValueError("minimum_average_stock_weight must be in [0, 1]")
        if self.maximum_turnover <= 0:
            raise ValueError("maximum_turnover must be positive")
        if min(
            self.stock_action_deadband,
            self.futures_action_deadband,
            self.option_coverage_deadband,
            self.stock_shortfall_score_penalty,
            self.turnover_score_penalty,
            self.put_ladder_drawdown_score_penalty,
        ) < 0:
            raise ValueError("PPO v2 deadbands and score penalties must be non-negative")


@dataclass(frozen=True)
class AppConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    option: OptionConfig = field(default_factory=OptionConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> "AppConfig":
        merged: dict[str, Any] = {}
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream) or {}
            _deep_update(merged, document)
        return cls.from_mapping(merged)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppConfig":
        """Build a config from a resolved run-metadata mapping."""
        merged = dict(data)
        market_data = dict(merged.get("market", {}))
        for regime_name in ("normal", "crisis"):
            if regime_name in market_data:
                market_data[regime_name] = RegimeParameters(**market_data[regime_name])
        if "transition_matrix" in market_data:
            market_data["transition_matrix"] = tuple(tuple(row) for row in market_data["transition_matrix"])
        if "seasonality_anchors" in market_data:
            market_data["seasonality_anchors"] = tuple(
                (str(month_day), float(value))
                for month_day, value in market_data["seasonality_anchors"]
            )
        baselines_data = dict(merged.get("baselines", {}))
        if "relative_momentum" in baselines_data:
            baselines_data["relative_momentum"] = RelativeMomentumConfig(
                **baselines_data["relative_momentum"]
            )
        if "rolling_hedge" in baselines_data:
            baselines_data["rolling_hedge"] = RollingHedgeConfig(
                **baselines_data["rolling_hedge"]
            )
        if "signed_rolling_hedge" in baselines_data:
            baselines_data["signed_rolling_hedge"] = SignedRollingHedgeConfig(
                **baselines_data["signed_rolling_hedge"]
            )
        return cls(
            market=MarketConfig(**market_data),
            portfolio=PortfolioConfig(**merged.get("portfolio", {})),
            option=OptionConfig(
                **{
                    **merged.get("option", {}),
                    **(
                        {"tenors": tuple(merged["option"]["tenors"])}
                        if "option" in merged and "tenors" in merged["option"]
                        else {}
                    ),
                }
            ),
            environment=EnvConfig(**merged.get("environment", {})),
            reward=RewardConfig(**merged.get("reward", {})),
            baselines=BaselinesConfig(**baselines_data),
            calibration=CalibrationConfig(**merged.get("calibration", {})),
            evaluation=EvaluationConfig(**merged.get("evaluation", {})),
            ppo=PPOConfig(
                **{
                    **merged.get("ppo", {}),
                    **(
                        {"policy_layers": tuple(merged["ppo"]["policy_layers"])}
                        if "ppo" in merged and "policy_layers" in merged["ppo"]
                        else {}
                    ),
                }
            ),
        )


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
