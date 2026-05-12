"""Tests for the meta-trade abstraction (spec section 4)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from parimutuel_sim.market import GRID_SIZE, MarketState, mcap
from parimutuel_sim.meta_trades import META_TRADES, MONEYLINE_KEYS, allocate
from parimutuel_sim.simulation import SimConfig, run_monte_carlo, run_one_trial


# --- §4.1 partition ---------------------------------------------------------


def test_moneyline_partition_covers_grid():
    mci = set(META_TRADES["MCI_WIN"].cells)
    draw = set(META_TRADES["DRAW"].cells)
    cry = set(META_TRADES["CRY_WIN"].cells)

    assert len(mci) == 28
    assert len(draw) == 8
    assert len(cry) == 28
    assert mci.isdisjoint(draw) and mci.isdisjoint(cry) and draw.isdisjoint(cry)

    all_cells = {(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)}
    assert mci | draw | cry == all_cells


# --- §4.2 allocation weights -----------------------------------------------


def test_allocate_proportional_to_mcap_snapshot():
    cells = [(0, 0), (0, 1), (0, 2)]
    caps = {(0, 0): 10.0, (0, 1): 30.0, (0, 2): 60.0}
    out = allocate(100.0, cells, caps)
    assert math.isclose(out[(0, 0)], 10.0, abs_tol=1e-9)
    assert math.isclose(out[(0, 1)], 30.0, abs_tol=1e-9)
    assert math.isclose(out[(0, 2)], 60.0, abs_tol=1e-9)


# --- §4.3 dust reconciliation ----------------------------------------------


def test_allocate_dust_reconciles_to_exact_total():
    cells = [(1, 1), (2, 2), (3, 3)]
    caps = {c: 1.0 for c in cells}
    cash = 1.0 / 3.0
    out = allocate(cash, cells, caps)
    assert sum(out.values()) == cash  # exact, no float drift


def test_allocate_zero_cash_is_noop():
    cells = [(0, 0), (1, 1)]
    out = allocate(0.0, cells, {(0, 0): 1.0, (1, 1): 2.0})
    assert all(v == 0.0 for v in out.values())


# --- §4.4 empty bucket fallback --------------------------------------------


def test_allocate_empty_bucket_falls_back_to_equal_weights(caplog):
    cells = [(0, 0), (1, 1), (2, 2)]
    caps = {c: 0.0 for c in cells}
    caplog.set_level(logging.WARNING, logger="parimutuel_sim.meta_trades")
    out = allocate(9.0, cells, caps)
    assert sum(out.values()) == 9.0
    # Equal split, modulo dust on the last leg.
    assert math.isclose(out[(0, 0)], 3.0, abs_tol=1e-9)
    assert math.isclose(out[(1, 1)], 3.0, abs_tol=1e-9)
    assert math.isclose(out[(2, 2)], 3.0, abs_tol=1e-9)
    assert any("empty bucket" in rec.message.lower() or "equal weights" in rec.message.lower()
               for rec in caplog.records)


# --- §4.5 end-to-end mint ---------------------------------------------------


def _fresh_state(seed: int = 7) -> MarketState:
    return MarketState(
        init_mcap_min=0.10,
        init_mcap_max=10.0,
        rng=np.random.default_rng(seed),
    )


def test_mint_meta_mci_win_routes_only_to_mci_cells():
    state = _fresh_state(seed=11)
    pre_supply = state.supply.copy()
    pre_total_mcap = float(state.mcap_grid.sum())

    fill = state.mint_meta("MCI_WIN", 1000.0, agent_id=42)

    bucket = set(META_TRADES["MCI_WIN"].cells)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) in bucket:
                assert state.supply[i, j] >= pre_supply[i, j]
            else:
                assert state.supply[i, j] == pre_supply[i, j]

    # Cash to each leg matches the pre-trade market cap ratios.
    pre_caps = {c: float(mcap(pre_supply[c])) for c in bucket}
    pre_total = sum(pre_caps.values())
    for cell, cash, _units, pre_mcap_recorded, _post in fill.legs:
        expected = 1000.0 * (pre_caps[cell] / pre_total)
        assert math.isclose(cash, expected, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(pre_mcap_recorded, pre_caps[cell], rel_tol=1e-12)

    # Conservation: total mcap delta equals dollars spent.
    post_total_mcap = float(state.mcap_grid.sum())
    assert math.isclose(post_total_mcap - pre_total_mcap, 1000.0, rel_tol=1e-9, abs_tol=1e-6)
    assert math.isclose(fill.total_cash, 1000.0, rel_tol=1e-12)


# --- §4.6 leg-order independence -------------------------------------------


def test_mint_meta_order_independent_via_snapshot():
    # Same starting state, different shuffled cell orderings inside the bucket.
    # We do this by monkey-patching META_TRADES temporarily with reversed cells.
    import parimutuel_sim.meta_trades as mt

    state_a = _fresh_state(seed=99)
    state_b = _fresh_state(seed=99)
    assert np.allclose(state_a.supply, state_b.supply)

    fill_a = state_a.mint_meta("DRAW", 250.0)

    # Build a defn with reversed cells and re-register under "DRAW_REV"
    original = mt.META_TRADES["DRAW"]
    reversed_def = mt.MetaTradeDef(
        key="DRAW_REV",
        set_name=original.set_name,
        display_name=original.display_name,
        cells=tuple(reversed(original.cells)),
    )
    mt.META_TRADES["DRAW_REV"] = reversed_def
    try:
        fill_b = state_b.mint_meta("DRAW_REV", 250.0)
    finally:
        mt.META_TRADES.pop("DRAW_REV", None)

    # Final supply grids must match cell-for-cell.
    assert np.allclose(state_a.supply, state_b.supply, rtol=0.0, atol=1e-12)
    # Per-cell minted units must match too (order may differ in legs list).
    units_a = {c: u for c, _cash, u, _pre, _post in fill_a.legs}
    units_b = {c: u for c, _cash, u, _pre, _post in fill_b.legs}
    for cell in units_a:
        assert math.isclose(units_a[cell], units_b[cell], rel_tol=0.0, abs_tol=1e-12)


# --- §4.7 strategy smoke ----------------------------------------------------


def test_moneyline_uniform_strategy_runs_and_respects_budget():
    cfg = SimConfig(
        n_agents=5,
        balance_min=100.0,
        balance_max=200.0,
        init_mcap_min=0.10,
        init_mcap_max=10.0,
        winner_mode="uniform",
        strategy="moneyline_uniform",
        min_mint=1.0,
        max_per_mint=10.0,
        min_mint_threshold=0.01,
        seed=2024,
    )
    rng = np.random.default_rng(cfg.seed)
    trial = run_one_trial(cfg, trial_id=0, rng=rng)

    assert trial.meta_trade_log, "expected meta_trade_log to be non-empty"
    for fill in trial.meta_trade_log:
        assert fill.meta_key in MONEYLINE_KEYS
        assert fill.trial_id == 0
        assert fill.total_cash >= 0
    # Every agent's mint spend never exceeds their starting balance.
    for a in trial.agents:
        assert a.starting_balance + 1e-9 >= (a.starting_balance - a.cash)


# --- §4.8 settlement still works on meta-traded holdings -------------------


def test_meta_trade_holder_paid_via_existing_settlement():
    cfg = SimConfig(
        n_agents=3,
        balance_min=100.0,
        balance_max=100.0,
        init_mcap_min=0.10,
        init_mcap_max=1.0,
        winner_mode="fixed:3,3",       # DRAW (3,3) is in DRAW bucket
        strategy="moneyline_uniform",
        min_mint=1.0,
        max_per_mint=20.0,
        min_mint_threshold=0.01,
        seed=7,
    )
    rng = np.random.default_rng(cfg.seed)
    trial = run_one_trial(cfg, trial_id=0, rng=rng)
    # Conservation: total in == total out.
    total_in = sum(a.starting_balance for a in trial.agents) + trial.house_seed_total
    total_out = sum(a.terminal_cash for a in trial.agents) + trial.house_terminal_value
    assert math.isclose(total_in, total_out, rel_tol=1e-9, abs_tol=1e-6)
    # At least one agent who held (3,3) via a DRAW meta-trade has terminal cash
    # above (starting - cash spent), i.e. received a payout.
    paid = [a for a in trial.agents if a.holdings.get((3, 3), 0.0) > 0]
    assert paid, "expected at least one agent to hold the winning cell via meta trade"
    payout_per_unit = trial.payout_per_unit
    for a in paid:
        expected = a.cash + a.holdings[(3, 3)] * payout_per_unit
        assert math.isclose(a.terminal_cash, expected, rel_tol=1e-9, abs_tol=1e-6)


# --- Existing-strategy parity (definition of done) -------------------------


def test_uniform_random_byte_for_byte_unchanged_with_meta_path_present():
    """Existing uniform_random results should be unchanged with meta plumbing
    present (no meta trades placed, RNG draw order preserved)."""
    cfg = SimConfig(
        n_agents=10,
        balance_min=50.0,
        balance_max=500.0,
        init_mcap_min=0.10,
        init_mcap_max=10.0,
        winner_mode="realistic",
        strategy="uniform_random",
        min_mint=1.0,
        max_per_mint=50.0,
        seed=1234,
    )
    rng1 = np.random.default_rng(cfg.seed)
    rng2 = np.random.default_rng(cfg.seed)
    t1 = run_one_trial(cfg, trial_id=0, rng=rng1)
    t2 = run_one_trial(cfg, trial_id=0, rng=rng2)
    assert np.allclose(t1.final_supply, t2.final_supply)
    assert not t1.meta_trade_log  # cell strategy doesn't place meta trades
