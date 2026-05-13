"""Run one trial of the market simulation and the outer Monte Carlo loop."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .agents import (
    CELL_STRATEGIES,
    META_STRATEGIES,
    MIXED_STRATEGY,
    CUSTOM_MIX_STRATEGY,
    Agent,
    assign_strategies,
    make_agents,
    pick_action,
)
from .market import GRID_SIZE, SEED_RNG_SALT, MarketState, marginal_payout, mcap, mint_units, marginal_price
from .meta_trades import (
    META_TRADES,
    allocate,
    MONEYLINE_KEYS,
    SPREAD_KEYS,
    TOTALS_KEYS,
    EXACT_SCORE_KEYS,
)
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
    winner_mode: str = "screenshot_table"
    # Default is now custom_mix with 80% moneyline, 10% totals, 5% spreads, 5% exact scores
    strategy: str = "custom_mix"
    min_mint: float = 1.0
    max_per_mint: float = 50.0
    min_mint_threshold: float = 0.01
    refund_on_empty: bool = True
    seed: int = 0
    log_events_for_first_k: int = 10
    meta_trades_enabled: bool = True
    meta_trade_mix: Optional[dict] = None  # prior used by moneyline_weighted
    # Mixed-cohort knobs. Only consulted when ``strategy == "mixed"``.
    meta_agent_fraction: float = 0.8
    cell_strategy: str = "uniform_random"
    meta_strategy: str = "moneyline_uniform"
    # Custom mix proportions. Used when ``strategy == "custom_mix"``.
    strategy_mix: Optional[dict] = field(default_factory=lambda: {
        "moneyline_weighted": 0.80,
        "totals_uniform": 0.10,
        "spreads_uniform": 0.05,
        "exact_score_weighted": 0.05,
    })

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
        if not (0.0 <= self.meta_agent_fraction <= 1.0):
            raise ValueError("meta_agent_fraction must be in [0, 1]")
        if self.strategy == CUSTOM_MIX_STRATEGY:
            if not self.strategy_mix:
                raise ValueError("strategy_mix must be provided for custom_mix strategy")
        elif self.strategy == MIXED_STRATEGY:
            if self.cell_strategy not in CELL_STRATEGIES:
                raise ValueError(
                    f"cell_strategy must be one of {CELL_STRATEGIES}, got {self.cell_strategy!r}"
                )
            if self.meta_strategy not in META_STRATEGIES:
                raise ValueError(
                    f"meta_strategy must be one of {META_STRATEGIES}, got {self.meta_strategy!r}"
                )
            if self.meta_agent_fraction > 0 and not self.meta_trades_enabled:
                raise ValueError(
                    "mixed strategy with meta_agent_fraction > 0 requires meta_trades_enabled=True"
                )


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
    meta_trade_log: list = field(default_factory=list)
    rejection_stats: dict = field(default_factory=dict)
    granular_stats: dict = field(default_factory=dict)
    moneyline_timeline: dict = field(default_factory=dict)

def categorize_action(action) -> str:
    if action.kind == "cell":
        return "Specific scores"
    k = action.meta_key
    if k in MONEYLINE_KEYS:
        return "Moneyline"
    if k in TOTALS_KEYS:
        return "Totals"
    if k in SPREAD_KEYS:
        return "Spreads"
    return "Specific scores"


def _step_agent(
    state: MarketState,
    agent: Agent,
    cfg: SimConfig,
    rng: np.random.Generator,
    tick: int,
) -> Optional[dict]:
    if agent.strategy in META_STRATEGIES and not cfg.meta_trades_enabled:
        raise ValueError(
            f"strategy {agent.strategy!r} requires meta_trades_enabled=True"
        )
    action = pick_action(
        state,
        agent.strategy,
        agent.cash,
        cfg.min_mint,
        cfg.max_per_mint,
        rng,
        meta_prior=cfg.meta_trade_mix,
    )
    if action.amount <= 0:
        return None

    pre_mcap_total = float(state.mcap_grid.sum())

    if action.kind == "cell":
        cell = action.cell
        pre_supply = float(state.supply[cell])
        post_supply, units = mint_units(pre_supply, action.amount)
        if units <= 0:
            return None
        post_mcap_total = pre_mcap_total + action.amount
        multiplier = (units / action.amount) * (post_mcap_total / post_supply)
        if multiplier <= 1.0:
            return {"rejected": True, "category": categorize_action(action)}

        units = state.mint(cell, action.amount)
        agent.cash -= action.amount
        agent.add_holding(cell, units)
        # Conservation invariant for the step: cash spent equals mcap delta.
        assert abs((float(state.mcap_grid.sum()) - pre_mcap_total) - action.amount) <= 1e-6 * max(
            1.0, action.amount
        ), "single-cell mint broke step conservation"
        return {
            "trial_id": None,
            "tick": tick,
            "kind": "cell",
            "agent_id": agent.agent_id,
            "cell_i": cell[0],
            "cell_j": cell[1],
            "meta_key": None,
            "mint_amt": action.amount,
            "units": units,
            "new_supply": float(state.supply[cell]),
            "marginal_payout": (4.0 / 7.0) * float(state.supply[cell]),
        }

    # Meta trade
    cells = META_TRADES[action.meta_key].cells
    mcaps = {c: float(state.mcap_grid[c]) for c in cells}
    alloc = allocate(action.amount, cells, mcaps)
    post_mcap_total = pre_mcap_total + action.amount
    passed = False
    for c, cash in alloc.items():
        if cash > 0:
            pre_supply = float(state.supply[c])
            post_supply, units = mint_units(pre_supply, cash)
            if units > 0:
                multiplier = (units / action.amount) * (post_mcap_total / post_supply)
                if multiplier > 1.0:
                    passed = True
                    break
    if not passed:
        return {"rejected": True, "category": categorize_action(action)}

    fill = state.mint_meta(action.meta_key, action.amount, agent_id=agent.agent_id)
    agent.cash -= action.amount
    for cell, _cash, units, _pre, _post in fill.legs:
        if units > 0:
            agent.add_holding(cell, units)
    assert abs((float(state.mcap_grid.sum()) - pre_mcap_total) - action.amount) <= 1e-6 * max(
        1.0, action.amount
    ), "meta-trade legs broke step conservation"
    fill.tick = tick
    return {
        "trial_id": None,
        "tick": tick,
        "kind": "meta",
        "agent_id": agent.agent_id,
        "cell_i": None,
        "cell_j": None,
        "meta_key": action.meta_key,
        "mint_amt": action.amount,
        "units": fill.total_units,
        "new_supply": None,
        "marginal_payout": None,
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
    assign_strategies(
        agents,
        cfg.strategy,
        rng,
        meta_agent_fraction=cfg.meta_agent_fraction,
        cell_strategy=cfg.cell_strategy,
        meta_strategy=cfg.meta_strategy,
        strategy_mix=cfg.strategy_mix,
    )
    attempts = {
        "Moneyline": 0, "Totals": 0, "Spreads": 0, "Specific scores": 0
    }
    rejections = {
        "Moneyline": 0, "Totals": 0, "Spreads": 0, "Specific scores": 0
    }
    
    attempts_granular = collections.defaultdict(int)
    rejections_granular = collections.defaultdict(int)

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
            rejection_stats={"attempts": attempts, "rejections": rejections},
            granular_stats={"attempts": attempts_granular, "rejections": rejections_granular},
        )

    event_log: Optional[List[dict]] = [] if log_events else None

    # Action loop: random agent acts each step until everyone is below threshold.
    active_ids = [a.agent_id for a in agents]
    by_id = {a.agent_id: a for a in agents}
    tick = 0
    consecutive_passes = 0
    max_passes = len(active_ids) * 5
    
    timeline = {"tick": [], "MCI_WIN": [], "DRAW": [], "CRY_WIN": []}
    mci_cells = META_TRADES["MCI_WIN"].cells
    draw_cells = META_TRADES["DRAW"].cells
    cry_cells = META_TRADES["CRY_WIN"].cells

    def log_timeline(current_tick):
        total_pool = float(state.mcap_grid.sum())
        timeline["tick"].append(current_tick)
        for key, cells in [("MCI_WIN", mci_cells), ("DRAW", draw_cells), ("CRY_WIN", cry_cells)]:
            p_bucket = sum(float(marginal_price(state.supply[c])) for c in cells)
            timeline[key].append(total_pool / p_bucket if p_bucket > 0 else 0.0)

    while active_ids:
        if tick % 5 == 0:
            log_timeline(tick)

        if consecutive_passes > max_passes:
            break
            
        idx = int(rng.integers(len(active_ids)))
        agent = by_id[active_ids[idx]]
        if agent.cash <= cfg.min_mint_threshold:
            active_ids.pop(idx)
            continue
            
        ev = _step_agent(state, agent, cfg, rng, tick=tick)
        if ev is not None:
            mkey = ev.get("meta_key") or "cell"
            if ev.get("rejected"):
                cat = ev["category"]
                attempts[cat] += 1
                rejections[cat] += 1
                attempts_granular[mkey] += 1
                rejections_granular[mkey] += 1
                consecutive_passes += 1
            else:
                if ev.get("kind") == "cell":
                    cat = "Specific scores"
                elif ev.get("meta_key") in MONEYLINE_KEYS:
                    cat = "Moneyline"
                elif ev.get("meta_key") in TOTALS_KEYS:
                    cat = "Totals"
                elif ev.get("meta_key") in SPREAD_KEYS:
                    cat = "Spreads"
                else:
                    cat = "Specific scores"
                attempts[cat] += 1
                attempts_granular[mkey] += 1
                
                tick += 1
                consecutive_passes = 0
                if event_log is not None:
                    ev["trial_id"] = trial_id
                    event_log.append(ev)
        else:
            # Should not happen anymore, but just in case
            consecutive_passes += 1
            
        if agent.cash <= cfg.min_mint_threshold:
            active_ids.pop(idx)

    # Stamp trial_id onto meta trade records (left None during the loop).
    for fill in state.meta_trade_log:
        fill.trial_id = trial_id

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
        meta_trade_log=list(state.meta_trade_log),
        rejection_stats={"attempts": attempts, "rejections": rejections},
        granular_stats={"attempts": attempts_granular, "rejections": rejections_granular},
        moneyline_timeline=timeline,
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
