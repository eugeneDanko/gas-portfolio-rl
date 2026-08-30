from __future__ import annotations

import numpy as np

from gas_portfolio_rl.simulation.models import Regime


def next_regime(
    current: Regime,
    transition_matrix: tuple[tuple[float, float], tuple[float, float]],
    rng: np.random.Generator,
) -> Regime:
    return Regime(int(rng.choice(2, p=transition_matrix[int(current)])))

