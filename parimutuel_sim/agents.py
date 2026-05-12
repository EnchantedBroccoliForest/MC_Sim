"""Agents that mint OTs against the bonding curve until their cash runs out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .market import GRID_SIZE, MarketState, marginal_payout
from .meta_trades import MONEYLINE_KEYS

META_STRATEGIES = ("moneyline_uniform", "moneyline_weighted")
CELL_STRATEGIES = ("uniform_random", "weighted_by_marginal_payout")
ALL_STRATEGIES = CELL_STRATEGIES + META_STRATEGIES


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


@dataclass(frozen=True)
class AgentAction:
    """What an agent wants to do this tick.

    ``kind`` is ``"cell"`` for a single-cell mint (``cell`` is set) or
    ``"meta"`` for a meta trade (``meta_key`` is set).
    """

    kind: str  # "cell" | "meta"
    amount: float
    cell: Optional[Tuple[int, int]] = None
    meta_key: Optional[str] = None


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


def pick_meta_key(
    strategy: str,
    rng: np.random.Generator,
    prior: Optional[Mapping[str, float]] = None,
) -> str:
    """Pick a moneyline meta-trade key per the configured strategy.

    ``prior`` is only consulted by ``moneyline_weighted`` and defaults to
    uniform. Future work can plug in a Poisson-derived prior here.
    """
    if strategy == "moneyline_uniform":
        return str(rng.choice(MONEYLINE_KEYS))
    if strategy == "moneyline_weighted":
        if prior is None:
            weights = np.full(len(MONEYLINE_KEYS), 1.0 / len(MONEYLINE_KEYS))
        else:
            weights = np.array([float(prior.get(k, 0.0)) for k in MONEYLINE_KEYS], dtype=float)
            s = weights.sum()
            if s <= 0:
                raise ValueError("moneyline_weighted prior has non-positive total")
            weights = weights / s
        return str(rng.choice(MONEYLINE_KEYS, p=weights))
    raise ValueError(f"Unknown meta strategy: {strategy}")


def pick_action(
    state: MarketState,
    strategy: str,
    cash: float,
    min_mint: float,
    max_per_mint: float,
    rng: np.random.Generator,
    meta_prior: Optional[Mapping[str, float]] = None,
) -> AgentAction:
    """Choose the next action (cell or meta) and the mint amount.

    For cell strategies the RNG draw order is ``pick_cell -> pick_mint_amount``
    to preserve byte-for-byte parity with the pre-meta-trade simulator.
    """
    if strategy in META_STRATEGIES:
        key = pick_meta_key(strategy, rng, meta_prior)
        amount = pick_mint_amount(cash, min_mint, max_per_mint, rng)
        return AgentAction(kind="meta", amount=amount, meta_key=key)
    cell = pick_cell(state, strategy, rng)
    amount = pick_mint_amount(cash, min_mint, max_per_mint, rng)
    return AgentAction(kind="cell", amount=amount, cell=cell)


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
