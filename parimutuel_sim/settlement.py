"""Winner selection and pro-rata settlement."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import poisson

from .market import GRID_SIZE, MarketState, mcap


RAW_SCREENSHOT_PROBS = np.array([
    [0.0809, 0.0472, 0.0288, 0.0148, 0.0062, 0.0018, 0.0013, 0.0018],
    [0.1416, 0.0954, 0.0394, 0.0176, 0.0080, 0.0018, 0.0029, 0.0018],
    [0.0864, 0.1183, 0.0363, 0.0112, 0.0044, 0.0023, 0.0003, 0.0003],
    [0.0443, 0.0529, 0.0335, 0.0073, 0.0008, 0.0003, 0.0005, 0.0005],
    [0.0187, 0.0241, 0.0132, 0.0023, 0.0021, 0.0000, 0.0000, 0.0000],
    [0.0054, 0.0054, 0.0070, 0.0008, 0.0000, 0.0000, 0.0003, 0.0000],
    [0.0039, 0.0086, 0.0008, 0.0016, 0.0000, 0.0008, 0.0000, 0.0003],
    [0.0054, 0.0054, 0.0008, 0.0016, 0.0000, 0.0008, 0.0000, 0.0000],
])
SCREENSHOT_PROBS = RAW_SCREENSHOT_PROBS / RAW_SCREENSHOT_PROBS.sum()


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
    if mode == "screenshot_table":
        return SCREENSHOT_PROBS.copy()
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
