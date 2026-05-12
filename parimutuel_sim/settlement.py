"""Winner selection and pro-rata settlement."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import poisson

from .market import GRID_SIZE, MarketState, mcap


def _truncated_poisson_probs(lam: float, k: int = GRID_SIZE) -> np.ndarray:
    """P(X=0), P(X=1), ..., P(X=k-2), P(X>=k-1). Sums to 1."""
    pmf = poisson.pmf(np.arange(k - 1), lam)
    tail = 1.0 - pmf.sum()
    return np.concatenate([pmf, [tail]])


def winner_probabilities(mode: str) -> np.ndarray:
    """Return the 8x8 prior over outcomes for the configured winner mode."""
    if mode == "uniform":
        return np.full((GRID_SIZE, GRID_SIZE), 1.0 / (GRID_SIZE * GRID_SIZE))
    if mode == "realistic":
        # Spec §6.1: independent Poisson with lambda_home=1.6, lambda_away=1.1
        ph = _truncated_poisson_probs(1.6)
        pa = _truncated_poisson_probs(1.1)
        grid = np.outer(ph, pa)
        grid /= grid.sum()
        return grid
    if mode.startswith("fixed:"):
        try:
            i_str, j_str = mode.split(":", 1)[1].split(",")
            i, j = int(i_str), int(j_str)
        except Exception as exc:
            raise ValueError(f"fixed mode requires format 'fixed:i,j', got {mode}") from exc
        if not (0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE):
            raise ValueError(f"fixed cell out of range: ({i},{j})")
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        grid[i, j] = 1.0
        return grid
    raise ValueError(f"Unknown winner mode: {mode}")


def draw_winner(probs: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    flat = probs.flatten()
    idx = int(rng.choice(flat.size, p=flat))
    return divmod(idx, GRID_SIZE)


def settle(
    state: MarketState,
    winner: Tuple[int, int],
    refund_on_empty: bool = True,
) -> Tuple[float, float]:
    """Return (total_pool, payout_per_unit). payout_per_unit == 0 means refund."""
    total_pool = float(state.mcap_grid.sum())
    supply_W = float(state.supply[winner])
    if supply_W <= 0:
        if refund_on_empty:
            return total_pool, 0.0
        raise ValueError(f"Winning OT {winner} has zero supply and refund disabled")
    payout_per_unit = total_pool / supply_W
    return total_pool, payout_per_unit
