"""Run one trial of the market simulation and the outer Monte Carlo loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .agents import Agent, make_agents, pick_cell, pick_mint_amount
from .market import GRID_SIZE, SEED_RNG_SALT, MarketState, marginal_payout, mcap
from .settlement import draw_winner, settle, winner_probabilities


@dataclass
class SimConfig:
    """All knobs for one Monte Carlo run."""

    n_agents: int = 100
    balance_min: float = 50.0
    balance_max: float = 5000.0
    n_trials: int = 1000
    init_mcap_min: float = 0.10
    init_mcap_max: float = 10.0
    winner_mode: str = "realistic"
    strategy: str = "uniform_random"
    min_mint: float = 1.0
    max_per_mint: float = 50.0
    min_mint_threshold: float = 0.01
    refund_on_empty: bool = True
    seed: int = 0
    log_events_for_first_k: int = 10

    def __post_init__(self):
        if self.n_agents < 0:
            raise ValueError("n_agents must be >= 0")
        if self.n_trials <= 0:
            raise ValueError("n_trials must be > 0")
        if self.balance_min < 0:
            raise ValueError("balance_min must be >= 0")
        if self.balance_max < self.balance_min:
            raise ValueError("balance_max must be >= balance_min")
        if self.init_mcap_min < 0 or self.init_mcap_max < 0:
            raise ValueError("init_mcap_min and init_mcap_max must be >= 0")
        if self.init_mcap_max < self.init_mcap_min:
            raise ValueError("init_mcap_max must be >= init_mcap_min")
        if self.min_mint < 0:
            raise ValueError("min_mint must be >= 0")
        if self.max_per_mint <= 0:
            raise ValueError("max_per_mint must be > 0")
        if self.min_mint_threshold < 0:
            raise ValueError("min_mint_threshold must be >= 0")
        if self.log_events_for_first_k < 0:
            raise ValueError("log_events_for_first_k must be >= 0")


@dataclass
class TrialResult:
    trial_id: int
    agents: List[Agent]
    final_supply: np.ndarray
    init_mcap_grid: np.ndarray
    init_supply_grid: np.ndarray
    n_holders: np.ndarray  # 8x8 int counts
    winner: Tuple[int, int]
    total_pool: float
    payout_per_unit: float
    house_seed_total: float
    house_terminal_value: float
    event_log: Optional[List[dict]] = None


def _step_agent(
    state: MarketState,
    agent: Agent,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Optional[dict]:
    cell = pick_cell(state, cfg.strategy, rng)
    mint_amt = pick_mint_amount(agent.cash, cfg.min_mint, cfg.max_per_mint, rng)
    if mint_amt <= 0:
        return None
    units = state.mint(cell, mint_amt)
    agent.cash -= mint_amt
    agent.add_holding(cell, units)
    return {
        "agent_id": agent.agent_id,
        "cell_i": cell[0],
        "cell_j": cell[1],
        "mint_amt": mint_amt,
        "units": units,
        "new_supply": float(state.supply[cell]),
        "marginal_payout": (4.0 / 7.0) * float(state.supply[cell]),
    }


def run_one_trial(
    cfg: SimConfig,
    trial_id: int,
    rng: np.random.Generator,
    log_events: bool = False,
) -> TrialResult:
    # Separate RNG stream for the initial market seed so that changing the
    # number of agents leaves the grid unchanged (spec §4.0).
    seed_rng = np.random.default_rng(int(cfg.seed) ^ SEED_RNG_SALT ^ int(trial_id))
    state = MarketState(
        init_mcap_min=cfg.init_mcap_min,
        init_mcap_max=cfg.init_mcap_max,
        rng=seed_rng,
    )
    init_mcap_grid = state.init_mcap.copy()
    init_supply_grid = state.init_supply.copy()
    house_seed_total = state.house_seed_total

    agents = make_agents(cfg.n_agents, cfg.balance_min, cfg.balance_max, rng)
    if not agents:
        # Degenerate: just settle on the seeded state.
        probs = winner_probabilities(cfg.winner_mode)
        winner = draw_winner(probs, rng)
        total_pool, payout_per_unit = settle(state, winner, cfg.refund_on_empty)
        house_terminal_value = float(state.init_supply[winner]) * payout_per_unit
        return TrialResult(
            trial_id=trial_id,
            agents=[],
            final_supply=state.supply.copy(),
            init_mcap_grid=init_mcap_grid,
            init_supply_grid=init_supply_grid,
            n_holders=np.zeros((GRID_SIZE, GRID_SIZE), dtype=int),
            winner=winner,
            total_pool=total_pool,
            payout_per_unit=payout_per_unit,
            house_seed_total=house_seed_total,
            house_terminal_value=house_terminal_value,
            event_log=[] if log_events else None,
        )

    event_log: Optional[List[dict]] = [] if log_events else None

    # Action loop: random agent acts each step until everyone is below threshold.
    active_ids = [a.agent_id for a in agents]
    by_id = {a.agent_id: a for a in agents}
    while active_ids:
        idx = int(rng.integers(len(active_ids)))
        agent = by_id[active_ids[idx]]
        if agent.cash <= cfg.min_mint_threshold:
            # Drop this agent and continue.
            active_ids.pop(idx)
            continue
        ev = _step_agent(state, agent, cfg, rng)
        if event_log is not None and ev is not None:
            ev["trial_id"] = trial_id
            event_log.append(ev)
        if agent.cash <= cfg.min_mint_threshold:
            active_ids.pop(idx)

    # Winner & settlement
    probs = winner_probabilities(cfg.winner_mode)
    winner = draw_winner(probs, rng)
    total_pool, payout_per_unit = settle(state, winner, cfg.refund_on_empty)

    # Compute terminal cash. Refund branch (payout_per_unit == 0) returns
    # each agent's mint spend at cost basis on losing OTs as well; in the
    # default seeded case this branch is unreachable.
    if payout_per_unit > 0:
        for a in agents:
            held = a.holdings.get(winner, 0.0)
            a.terminal_cash = a.cash + held * payout_per_unit
        house_terminal_value = float(state.init_supply[winner]) * payout_per_unit
    else:
        # Refund mode: agents get all spent dollars back. Their cost basis on each
        # cell equals mcap_delta = mcap(x_after_their_share) − mcap(before). We
        # can't reconstruct that exactly without tracking it, but in refund mode
        # we know total_pool equals total agent spend + house seed; so each agent
        # receives back exactly (starting_balance - cash) for fairness, and the
        # house gets back its seed.
        for a in agents:
            spent = a.starting_balance - a.cash
            a.terminal_cash = a.cash + spent  # == starting_balance
        house_terminal_value = house_seed_total

    # Holder counts per cell
    n_holders = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for a in agents:
        for (i, j), units in a.holdings.items():
            if units > 0:
                n_holders[i, j] += 1

    return TrialResult(
        trial_id=trial_id,
        agents=agents,
        final_supply=state.supply.copy(),
        init_mcap_grid=init_mcap_grid,
        init_supply_grid=init_supply_grid,
        n_holders=n_holders,
        winner=winner,
        total_pool=total_pool,
        payout_per_unit=payout_per_unit,
        house_seed_total=house_seed_total,
        house_terminal_value=house_terminal_value,
        event_log=event_log,
    )


def run_monte_carlo(cfg: SimConfig, verbose: bool = True):
    """Run S trials. Yields TrialResult per trial (generator for memory)."""
    rng = np.random.default_rng(cfg.seed)
    for trial_id in range(cfg.n_trials):
        # Each trial gets its own independent RNG stream for the agent loop.
        sub_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        log_events = trial_id < cfg.log_events_for_first_k
        result = run_one_trial(cfg, trial_id, sub_rng, log_events=log_events)
        if verbose and (trial_id + 1) % max(1, cfg.n_trials // 20) == 0:
            print(f"  trial {trial_id + 1}/{cfg.n_trials} done")
        yield result
