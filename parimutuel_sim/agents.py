"""Agents that mint OTs against the bonding curve until their cash runs out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .market import GRID_SIZE, MarketState, marginal_payout
from .meta_trades import MONEYLINE_KEYS, SPREAD_KEYS, TOTALS_KEYS, EXACT_SCORE_KEYS
from .settlement import SCREENSHOT_PROBS

META_STRATEGIES = (
    "moneyline_uniform",
    "moneyline_weighted",
    "spreads_uniform",
    "totals_uniform",
    "exact_score_uniform",
    "exact_score_weighted",
)
CELL_STRATEGIES = ("uniform_random", "weighted_by_marginal_payout")
MIXED_STRATEGY = "mixed"
CUSTOM_MIX_STRATEGY = "custom_mix"
ALL_STRATEGIES = CELL_STRATEGIES + META_STRATEGIES + (MIXED_STRATEGY, CUSTOM_MIX_STRATEGY)


@dataclass
class Agent:
    agent_id: int
    starting_balance: float
    cash: float
    holdings: Dict[Tuple[int, int], float] = field(default_factory=dict)
    terminal_cash: float = 0.0
    # Per-agent strategy. Assigned after construction (see assign_strategies).
    # Always one of CELL_STRATEGIES | META_STRATEGIES — never "mixed".
    strategy: str = ""

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
    if strategy == "spreads_uniform":
        return str(rng.choice(SPREAD_KEYS))
    if strategy == "totals_uniform":
        return str(rng.choice(TOTALS_KEYS))
    if strategy == "exact_score_uniform":
        return str(rng.choice(EXACT_SCORE_KEYS))
    if strategy == "exact_score_weighted":
        flat_probs = SCREENSHOT_PROBS.flatten()
        idx = int(rng.choice(len(EXACT_SCORE_KEYS), p=flat_probs))
        return EXACT_SCORE_KEYS[idx]
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


def assign_strategies(
    agents: list[Agent],
    strategy: str,
    rng: np.random.Generator,
    meta_agent_fraction: float = 0.8,
    cell_strategy: str = "uniform_random",
    meta_strategy: str = "moneyline_uniform",
    strategy_mix: Optional[Mapping[str, float]] = None,
) -> None:
    """Set ``agent.strategy`` on every agent.

    For non-mixed strategies all agents share the configured value and no RNG
    is consumed (preserving byte-for-byte parity with single-strategy runs).
    For ``strategy == "mixed"``, each agent independently picks the meta
    sub-strategy with probability ``meta_agent_fraction`` and the cell
    sub-strategy otherwise.
    For ``strategy == "custom_mix"``, it uses ``strategy_mix``.
    """
    if strategy == CUSTOM_MIX_STRATEGY:
        if strategy_mix is None:
            raise ValueError("strategy_mix must be provided for custom_mix strategy")
        strategies = list(strategy_mix.keys())
        probs = list(strategy_mix.values())
        if abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("strategy_mix probabilities must sum to 1")
        picks = rng.choice(strategies, p=probs, size=len(agents))
        for a, s in zip(agents, picks):
            a.strategy = str(s)
        return
    if strategy == MIXED_STRATEGY:
        if cell_strategy not in CELL_STRATEGIES:
            raise ValueError(f"cell_strategy must be one of {CELL_STRATEGIES}, got {cell_strategy!r}")
        if meta_strategy not in META_STRATEGIES:
            raise ValueError(f"meta_strategy must be one of {META_STRATEGIES}, got {meta_strategy!r}")
        if not (0.0 <= meta_agent_fraction <= 1.0):
            raise ValueError(f"meta_agent_fraction must be in [0, 1], got {meta_agent_fraction}")
        if not agents:
            return
        picks = rng.random(size=len(agents)) < meta_agent_fraction
        for a, is_meta in zip(agents, picks):
            a.strategy = meta_strategy if is_meta else cell_strategy
        return
    if strategy not in CELL_STRATEGIES and strategy not in META_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}")
    for a in agents:
        a.strategy = strategy
