"""Unit tests for the parimutuel market (spec section 10)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from parimutuel_sim.analytics import AGENT_PNL_COLUMNS, build_agent_pnl_df
from parimutuel_sim.market import (
    GRID_SIZE,
    PRICE_SCALE,
    MarketState,
    cost_to_mint,
    marginal_payout,
    marginal_price,
    mcap,
    mint_units,
    supply_for_mcap,
)
from parimutuel_sim.settlement import settle
from parimutuel_sim.simulation import SimConfig, run_monte_carlo, run_one_trial
from parimutuel_sim.viz import plot_pnl_histogram, plot_roi_vs_balance


def test_mint_from_zero():
    """Minting $D from x=0 yields x = ((7/4)*PRICE_SCALE*D)^(4/7) and mcap(x) ≈ D."""
    D = 12.34
    x_new, units = mint_units(0.0, D)
    expected_x = ((7.0 / 4.0) * PRICE_SCALE * D) ** (4.0 / 7.0)
    assert math.isclose(x_new, expected_x, rel_tol=1e-12)
    assert math.isclose(units, expected_x, rel_tol=1e-12)
    assert math.isclose(mcap(x_new), D, rel_tol=1e-9, abs_tol=1e-12)


def test_mint_additive():
    """Minting $A then $B from same start equals minting $(A+B) once."""
    x0 = 5.0
    A, B = 7.0, 13.0
    x_a, _ = mint_units(x0, A)
    x_ab, _ = mint_units(x_a, B)
    x_once, _ = mint_units(x0, A + B)
    assert math.isclose(x_ab, x_once, rel_tol=1e-12)


def test_marginal_price_monotone():
    """p(x) = x^(3/4) is strictly increasing."""
    xs = np.linspace(0.0, 100.0, 200)
    ps = marginal_price(xs)
    diffs = np.diff(ps)
    assert np.all(diffs >= 0)
    # Strictly increasing for x > 0
    assert np.all(diffs[1:] > 0)


def test_marginal_payout_formula():
    """marginal_payout(x) == (4/7) * x."""
    for x in [0.0, 1.0, 7.5, 123.456]:
        assert math.isclose(marginal_payout(x), (4.0 / 7.0) * x, rel_tol=1e-12)


def test_conservation_with_house():
    """Total cash + house value at settlement equals starting cash + house seed."""
    cfg = SimConfig(
        n_agents=20,
        balance_min=100.0,
        balance_max=500.0,
        init_mcap_min=0.10,
        init_mcap_max=10.0,
        winner_mode="uniform",
        strategy="uniform_random",
        min_mint=1.0,
        max_per_mint=50.0,
        min_mint_threshold=0.01,
        seed=42,
    )
    rng = np.random.default_rng(cfg.seed)
    trial = run_one_trial(cfg, trial_id=0, rng=rng)
    total_in = sum(a.starting_balance for a in trial.agents) + trial.house_seed_total
    total_out = (
        sum(a.terminal_cash for a in trial.agents) + trial.house_terminal_value
    )
    # Conservation: every dollar in == every dollar out (within float tolerance)
    assert math.isclose(total_in, total_out, rel_tol=1e-9, abs_tol=1e-6)


def test_pro_rata_settlement():
    """Two agents with equal units of the winning OT split the pool, no house seed."""
    state = MarketState(init_mcap_min=0.0, init_mcap_max=0.0, rng=np.random.default_rng(0))
    winner = (1, 1)
    other = (3, 4)
    units_each = 10.0
    state.supply[winner] = 2 * units_each
    state.supply[other], _ = mint_units(0.0, 50.0)
    total_pool, payout_per_unit = settle(state, winner)
    payout_each = units_each * payout_per_unit
    assert math.isclose(payout_each, total_pool / 2.0, rel_tol=1e-12)


def test_seed_reproducibility(tmp_path):
    """Two runs with same seed produce identical results."""
    cfg = SimConfig(
        n_agents=15,
        balance_min=100.0,
        balance_max=500.0,
        init_mcap_min=0.10,
        init_mcap_max=10.0,
        winner_mode="uniform",
        strategy="uniform_random",
        min_mint=1.0,
        max_per_mint=50.0,
        seed=12345,
    )
    rng1 = np.random.default_rng(cfg.seed)
    rng2 = np.random.default_rng(cfg.seed)
    t1 = run_one_trial(cfg, trial_id=0, rng=rng1)
    t2 = run_one_trial(cfg, trial_id=0, rng=rng2)
    # Initial mcap grids should be identical
    assert np.allclose(t1.init_mcap_grid, t2.init_mcap_grid)
    # Agent terminal cash should be identical
    pnl1 = sorted((a.agent_id, a.terminal_cash) for a in t1.agents)
    pnl2 = sorted((a.agent_id, a.terminal_cash) for a in t2.agents)
    for (id1, c1), (id2, c2) in zip(pnl1, pnl2):
        assert id1 == id2
        assert math.isclose(c1, c2, rel_tol=1e-12)


def test_init_seed_range():
    """After seeding, every cell has mcap in [min, max] and x > 0."""
    init_min, init_max = 0.10, 10.0
    rng = np.random.default_rng(7)
    state = MarketState(init_mcap_min=init_min, init_mcap_max=init_max, rng=rng)
    assert state.supply.shape == (GRID_SIZE, GRID_SIZE)
    assert np.all(state.supply > 0)
    mcaps = (4.0 / 7.0) * state.supply ** (7.0 / 4.0) / PRICE_SCALE
    assert np.all(mcaps >= init_min - 1e-12)
    assert np.all(mcaps <= init_max + 1e-12)


def test_market_state_cached_mcap_tracks_mints():
    """The cached total mcap stays aligned with the bonding curve state."""
    state = MarketState(init_mcap_min=0.10, init_mcap_max=10.0, rng=np.random.default_rng(7))
    state.mint((2, 3), 12.5)
    state.mint_meta("DRAW", 25.0)

    assert math.isclose(
        state.total_mcap,
        float(mcap(state.supply).sum()),
        rel_tol=1e-12,
        abs_tol=1e-9,
    )


def test_market_state_cached_mcap_tracks_many_mints():
    """The cached total mcap does not drift materially over a long run."""
    rng = np.random.default_rng(123)
    state = MarketState(init_mcap_min=0.10, init_mcap_max=10.0, rng=np.random.default_rng(7))

    for _ in range(10_000):
        cell = (int(rng.integers(GRID_SIZE)), int(rng.integers(GRID_SIZE)))
        state.mint(cell, float(rng.uniform(0.01, 50.0)))

    assert math.isclose(
        state.total_mcap,
        float(mcap(state.supply).sum()),
        rel_tol=1e-10,
        abs_tol=1e-7,
    )


def test_seed_independence():
    """Changing N (agent count) leaves the seeded initial grid unchanged."""
    seed = 9999
    cfg_a = SimConfig(n_agents=10, seed=seed)
    cfg_b = SimConfig(n_agents=200, seed=seed)
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)
    t_a = run_one_trial(cfg_a, trial_id=0, rng=rng_a)
    t_b = run_one_trial(cfg_b, trial_id=0, rng=rng_b)
    assert np.allclose(t_a.init_mcap_grid, t_b.init_mcap_grid)


def test_supply_for_mcap_inverse():
    """supply_for_mcap is the inverse of mcap."""
    for m in [0.001, 0.1, 1.0, 100.0]:
        x = supply_for_mcap(m)
        assert math.isclose(mcap(x), m, rel_tol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_per_mint": 0.0},
        {"max_per_mint": -1.0},
        {"min_mint_threshold": -0.01},
        {"balance_min": -1.0},
        {"balance_min": 10.0, "balance_max": 1.0},
        {"n_trials": 0},
    ],
)
def test_invalid_sim_config_rejected(kwargs):
    """Invalid configs fail fast instead of entering the mint loop."""
    with pytest.raises(ValueError):
        SimConfig(**kwargs)


def test_negative_seed_range_rejected():
    """Negative seed market caps would create NaN supplies."""
    with pytest.raises(ValueError):
        MarketState(init_mcap_min=-1.0, init_mcap_max=1.0)


def test_zero_agent_pnl_dataframe_has_schema():
    """Zero-agent trials still produce an empty table with expected columns."""
    cfg = SimConfig(n_agents=0, n_trials=1, seed=123)
    trials = list(run_monte_carlo(cfg, verbose=False))
    agent_df = build_agent_pnl_df(trials)

    assert agent_df.empty
    assert list(agent_df.columns) == AGENT_PNL_COLUMNS


def test_plots_handle_constant_starting_balance(tmp_path):
    """Constant balances should not crash qcut/cut-based plots."""
    cfg = SimConfig(n_agents=5, balance_min=100.0, balance_max=100.0, n_trials=1, seed=123)
    trials = list(run_monte_carlo(cfg, verbose=False))
    agent_df = build_agent_pnl_df(trials)

    pnl_out = tmp_path / "pnl.png"
    roi_out = tmp_path / "roi.png"
    plot_pnl_histogram(agent_df, pnl_out)
    plot_roi_vs_balance(agent_df, roi_out)

    assert pnl_out.exists()
    assert roi_out.exists()
