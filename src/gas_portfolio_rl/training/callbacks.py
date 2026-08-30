from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from gas_portfolio_rl.baselines import ProtectivePutBaseline, run_strategy
from gas_portfolio_rl.config.models import AppConfig
from gas_portfolio_rl.environments.gas_portfolio_env import GasPortfolioEnv
from gas_portfolio_rl.evaluation.metrics import episode_metrics
from gas_portfolio_rl.evaluation.policy import NormalizedModelStrategy
from gas_portfolio_rl.evaluation.scenarios import make_scenarios
from gas_portfolio_rl.visualization.ppo_diagnostics import plot_training_diagnostics


class RiskValidationCallback(BaseCallback):
    """Deterministic validation with fatal-event constrained model selection."""

    def __init__(
        self,
        config: AppConfig,
        output_dir: str | Path,
        *,
        seed_start: int = 10_000,
        stress_seed_start: int = 30_000,
        live_display: bool = False,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.config = config
        self.output_dir = Path(output_dir)
        self.seed_start = seed_start
        self.stress_seed_start = stress_seed_start
        self.best_score = -np.inf
        self.live_display = live_display
        self._display_handle = None
        self._next_evaluation = config.ppo.eval_frequency
        self._evaluations = 0
        self._crisis_put_ladder_drawdown: float | None = None

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_evaluation:
            self._evaluate_and_save()
            self._next_evaluation += self.config.ppo.eval_frequency
        return True

    def _on_training_end(self) -> None:
        if self._evaluations == 0 or self.num_timesteps < self._next_evaluation - self.config.ppo.eval_frequency:
            self._evaluate_and_save()

    def _evaluate_and_save(self) -> None:
        normalizer = self.model.get_vec_normalize_env()
        if normalizer is None:
            raise RuntimeError("RiskValidationCallback requires VecNormalize")
        scenarios = make_scenarios(self.config)
        rows = self._evaluate_scenario(
            scenarios["base"],
            normalizer,
            range(self.seed_start, self.seed_start + self.config.ppo.validation_episodes),
        )
        stress_rows = self._evaluate_scenario(
            scenarios["crisis_stress"],
            normalizer,
            range(
                self.stress_seed_start,
                self.stress_seed_start + self.config.ppo.validation_stress_episodes,
            ),
        )

        base_margin_rate = float(np.mean([int(row["margin_calls"]) > 0 for row in rows]))
        stress_margin_rate = float(
            np.mean([int(row["margin_calls"]) > 0 for row in stress_rows])
        )
        margin_rate = max(base_margin_rate, stress_margin_rate)
        base_bankruptcy_rate = float(np.mean([bool(row["bankrupt"]) for row in rows]))
        stress_bankruptcy_rate = float(
            np.mean([bool(row["bankrupt"]) for row in stress_rows])
        )
        bankruptcy_rate = max(base_bankruptcy_rate, stress_bankruptcy_rate)
        mean_reward = float(np.mean([float(row["total_reward"]) for row in rows]))
        crisis_mean_reward = float(
            np.mean([float(row["total_reward"]) for row in stress_rows])
        )
        weight = self.config.ppo.crisis_validation_weight
        robust_reward = (1.0 - weight) * mean_reward + weight * crisis_mean_reward
        high_cash_fraction = float(np.mean([row["high_cash_fraction"] for row in rows]))
        high_cash_excess = max(
            0.0,
            high_cash_fraction - self.config.ppo.maximum_high_cash_fraction,
        )
        mean_stock_weight = float(
            np.mean([float(row["average_stock_weight"]) for row in rows])
        )
        mean_turnover = float(np.mean([float(row["turnover"]) for row in rows]))
        mean_option_coverage = float(
            np.mean([float(row.get("average_option_coverage", 0.0)) for row in rows])
        )
        crisis_mean_option_coverage = float(
            np.mean(
                [float(row.get("average_option_coverage", 0.0)) for row in stress_rows]
            )
        )
        stock_shortfall = max(
            0.0,
            self.config.ppo.minimum_average_stock_weight - mean_stock_weight,
        )
        turnover_excess = max(
            0.0,
            mean_turnover - self.config.ppo.maximum_turnover,
        )
        crisis_mean_max_drawdown = float(
            np.mean([row["max_drawdown"] for row in stress_rows])
        )
        if self._crisis_put_ladder_drawdown is None:
            self._crisis_put_ladder_drawdown = self._evaluate_put_ladder_drawdown(
                scenarios["crisis_stress"],
                range(
                    self.stress_seed_start,
                    self.stress_seed_start + self.config.ppo.validation_stress_episodes,
                ),
            )
        put_ladder_drawdown_excess = max(
            0.0, crisis_mean_max_drawdown - self._crisis_put_ladder_drawdown
        )
        margin_call_score_cost = self.config.ppo.margin_call_score_penalty * margin_rate
        bankruptcy_score_cost = (
            self.config.ppo.bankruptcy_score_penalty * bankruptcy_rate
        )
        high_cash_score_cost = self.config.ppo.high_cash_score_penalty * high_cash_excess
        stock_shortfall_score_cost = (
            self.config.ppo.stock_shortfall_score_penalty * stock_shortfall
        )
        turnover_score_cost = self.config.ppo.turnover_score_penalty * turnover_excess
        put_ladder_drawdown_score_cost = (
            self.config.ppo.put_ladder_drawdown_score_penalty
            * put_ladder_drawdown_excess
        )
        score = robust_reward - sum(
            (
                margin_call_score_cost,
                bankruptcy_score_cost,
                high_cash_score_cost,
                stock_shortfall_score_cost,
                turnover_score_cost,
                put_ladder_drawdown_score_cost,
            )
        )
        record = {
            "timesteps": self.num_timesteps,
            "score": score,
            "robust_reward": robust_reward,
            "mean_reward": mean_reward,
            "crisis_mean_reward": crisis_mean_reward,
            "mean_total_return": float(np.mean([row["total_return"] for row in rows])),
            "crisis_mean_total_return": float(
                np.mean([row["total_return"] for row in stress_rows])
            ),
            "mean_max_drawdown": float(np.mean([row["max_drawdown"] for row in rows])),
            "crisis_mean_max_drawdown": crisis_mean_max_drawdown,
            "crisis_put_ladder_mean_max_drawdown": self._crisis_put_ladder_drawdown,
            "put_ladder_drawdown_excess": put_ladder_drawdown_excess,
            "margin_call_score_cost": margin_call_score_cost,
            "bankruptcy_score_cost": bankruptcy_score_cost,
            "high_cash_score_cost": high_cash_score_cost,
            "stock_shortfall_score_cost": stock_shortfall_score_cost,
            "turnover_score_cost": turnover_score_cost,
            "put_ladder_drawdown_score_cost": put_ladder_drawdown_score_cost,
            "mean_cvar": float(np.mean([row["cvar"] for row in rows])),
            "crisis_mean_cvar": float(np.mean([row["cvar"] for row in stress_rows])),
            "margin_call_rate": margin_rate,
            "base_margin_call_rate": base_margin_rate,
            "crisis_margin_call_rate": stress_margin_rate,
            "bankruptcy_rate": bankruptcy_rate,
            "base_bankruptcy_rate": base_bankruptcy_rate,
            "crisis_bankruptcy_rate": stress_bankruptcy_rate,
            "mean_cash_weight": float(np.mean([row["average_cash_weight"] for row in rows])),
            "high_cash_fraction": high_cash_fraction,
            "high_cash_excess": high_cash_excess,
            "mean_stock_weight": mean_stock_weight,
            "stock_shortfall": stock_shortfall,
            "mean_turnover": mean_turnover,
            "turnover_excess": turnover_excess,
            "mean_option_coverage": mean_option_coverage,
            "crisis_mean_option_coverage": crisis_mean_option_coverage,
            "mean_short_put_coverage": float(
                np.mean([float(row.get("average_short_put_coverage", 0.0)) for row in rows])
            ),
            "mean_medium_put_coverage": float(
                np.mean([float(row.get("average_medium_put_coverage", 0.0)) for row in rows])
            ),
            "mean_long_put_coverage": float(
                np.mean([float(row.get("average_long_put_coverage", 0.0)) for row in rows])
            ),
        }
        self._append_record(record)
        self._evaluations += 1
        self._refresh_live_display()
        for key, value in record.items():
            if key != "timesteps":
                self.logger.record(f"validation/{key}", value)
        if self.verbose:
            print(
                f"validation steps={self.num_timesteps} score={score:.4f} "
                f"base_reward={mean_reward:.4f} crisis_reward={crisis_mean_reward:.4f} "
                f"base_return={record['mean_total_return']:.2%} "
                f"crisis_return={record['crisis_mean_total_return']:.2%} "
                f"MC={margin_rate:.1%}"
            )
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.output_dir / "best_model")
            normalizer.save(self.output_dir / "best_vecnormalize.pkl")

    def _evaluate_scenario(self, scenario_config, normalizer, seeds):
        rows = []
        for seed in seeds:
            strategy = NormalizedModelStrategy(
                self.model,
                normalizer,
                ppo_config=self.config.ppo,
                name="ppo",
            )
            run = run_strategy(GasPortfolioEnv(scenario_config), strategy, seed=seed)
            row = episode_metrics(
                run,
                initial_value=scenario_config.portfolio.initial_cash,
                trading_days=scenario_config.market.trading_days,
                annual_cash_rate=scenario_config.portfolio.annual_cash_rate,
                cvar_alpha=scenario_config.evaluation.cvar_alpha,
            )
            trajectory = run.trajectory[1:]
            row["average_cash_weight"] = float(
                np.mean([float(item["cash_weight"]) for item in trajectory])
            )
            row["high_cash_fraction"] = float(
                np.mean([float(item["cash_weight"]) > 0.80 for item in trajectory])
            )
            rows.append(row)
        return rows

    def _evaluate_put_ladder_drawdown(self, scenario_config, seeds) -> float:
        drawdowns = []
        for seed in seeds:
            strategy = ProtectivePutBaseline(
                stock_weight=self.config.ppo.minimum_stock_weight,
                tenor_allocation=(1.0, 1.0, 1.0),
                name="minimum_stock_put_ladder",
            )
            run = run_strategy(GasPortfolioEnv(scenario_config), strategy, seed=seed)
            row = episode_metrics(
                run,
                initial_value=scenario_config.portfolio.initial_cash,
                trading_days=scenario_config.market.trading_days,
                annual_cash_rate=scenario_config.portfolio.annual_cash_rate,
                cvar_alpha=scenario_config.evaluation.cvar_alpha,
            )
            drawdowns.append(float(row["max_drawdown"]))
        return float(np.mean(drawdowns))

    def _refresh_live_display(self) -> None:
        figure = plot_training_diagnostics(self.output_dir, save=True)
        if not self.live_display:
            return
        try:
            from IPython.display import display

            if self._display_handle is None:
                self._display_handle = display(figure, display_id=True)
            else:
                self._display_handle.update(figure)
        except (ImportError, AttributeError):
            # Saving the PNG is the portable fallback outside a notebook.
            self.live_display = False

    def _append_record(self, record: dict[str, float | int]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "validation_metrics.csv"
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)
