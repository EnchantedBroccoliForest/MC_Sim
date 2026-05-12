"""Agents that mint OTs against the bonding curve until their cash runs out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .market import GRID_SIZE, MarketState, marginal_payout


@dataclass
class Agent:
    agent_id: int
    starting_balance: float
    cash: float
    holdings: Dict[Tuple[int, int], float] = field(default_factory=dict)
    terminal_cash: float = 0.0

    def add_holding(self, cell: Tuple[int, int], units: float) -> None:
        self.holdings[cell] = self.holdings.get(cell, 0.0) + units

    @property
    def cells_held(self) -> int:
        return sum(1 for v in self.holdings.values() if v > 0)


def pick_cell(
    state: MarketState,
    strategy: str,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Pick a cell per the configured strategy."""
    if strategy == "uniform_random":
        i = int(rng.integers(GRID_SIZE))
        j = int(rng.integers(GRID_SIZE))
        return i, j
    if strategy == "weighted_by_marginal_payout":
        # Higher marginal payout (= more supply already) gets more weight.
        # Add a small floor so empty cells still have a chance.
        weights = marginal_payout(state.supply).flatten() + 1e-6
        weights = weights / weights.sum()
        idx = int(rng.choice(weights.size, p=weights))
        return divmod(idx, GRID_SIZE)
    raise ValueError(f"Unknown strategy: {strategy}")


def pick_mint_amount(
    cash: float,
    min_mint: float,
    max_per_mint: float,
    rng: np.random.Generator,
) -> float:
    """Pick a USD mint amount uniformly within configured bounds."""
    upper = min(cash, max_per_mint)
    if upper <= min_mint:
        return upper
    return float(rng.uniform(min_mint, upper))


def make_agents(
    n_agents: int,
    balance_min: float,
    balance_max: float,
    rng: np.random.Generator,
) -> list[Agent]:
    balances = rng.uniform(balance_min, balance_max, size=n_agents)
    return [
        Agent(agent_id=i, starting_balance=float(b), cash=float(b))
        for i, b in enumerate(balances)
    ]
