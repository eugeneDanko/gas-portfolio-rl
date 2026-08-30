from gas_portfolio_rl.training.seed_wrapper import EpisodeSeedWrapper
from gas_portfolio_rl.training.action_wrapper import (
    PPOPortfolioActionWrapper,
    apply_action_deadband,
    enforce_hedge_constraints,
    map_policy_action,
    smooth_option_targets,
    wrap_ppo_env,
)

__all__ = [
    "EpisodeSeedWrapper",
    "PPOPortfolioActionWrapper",
    "apply_action_deadband",
    "enforce_hedge_constraints",
    "map_policy_action",
    "smooth_option_targets",
    "wrap_ppo_env",
    "TrainingResult",
    "train_ppo",
]


def __getattr__(name: str):
    if name in {"TrainingResult", "train_ppo"}:
        from gas_portfolio_rl.training.trainer import TrainingResult, train_ppo

        return {"TrainingResult": TrainingResult, "train_ppo": train_ppo}[name]
    raise AttributeError(name)
