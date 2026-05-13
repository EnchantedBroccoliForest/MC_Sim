"""Tests for the mixed cohort (80% meta / 20% cell by default) behavior."""

from __future__ import annotations

import math

import numpy as np
import pytest

from parimutuel_sim.agents import (
    CELL_STRATEGIES,
    META_STRATEGIES,
    Agent,
    assign_strategies,
)
from parimutuel_sim.simulation import SimConfig, run_one_trial


def _mk_agents(n: int) -> list[Agent]:
    return [Agent(agent_id=i, starting_balance=100.0, cash=100.0) for i in range(n)]


def test_default_strategy_is_custom_mix():
    cfg = SimConfig()
    assert cfg.strategy == "custom_mix"
    assert cfg.strategy_mix == {
        "moneyline_weighted": 0.80,
        "totals_uniform": 0.10,
        "spreads_uniform": 0.05,
        "exact_score_weighted": 0.05,
    }
    assert cfg.meta_agent_fraction == 0.8
    assert cfg.cell_strategy == "uniform_random"
    assert cfg.meta_strategy == "moneyline_uniform"


def test_assign_strategies_mixed_respects_fraction():
    n = 2000
    agents = _mk_agents(n)
    rng = np.random.default_rng(42)
    assign_strategies(
        agents,
        "mixed",
        rng,
        meta_agent_fraction=0.8,
        cell_strategy="uniform_random",
        meta_strategy="moneyline_uniform",
    )
    n_meta = sum(1 for a in agents if a.strategy in META_STRATEGIES)
    n_cell = sum(1 for a in agents if a.strategy in CELL_STRATEGIES)
    # Every agent is assigned exactly one strategy.
    assert n_meta + n_cell == n
    # Empirical fraction should be near 0.8 (Bernoulli(0.8) over 2000 draws,
    # stddev ≈ 0.009, allow 4σ tolerance).
    assert abs(n_meta / n - 0.8) < 0.04


def test_assign_strategies_single_mode_uses_no_rng():
    """Non-mixed strategies must not consume RNG draws (parity guarantee)."""
    agents_a = _mk_agents(50)
    agents_b = _mk_agents(50)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    assign_strategies(agents_a, "uniform_random", rng_a)
    assign_strategies(agents_b, "moneyline_uniform", rng_b)
    # Both RNGs untouched, so a follow-up draw should match.
    assert rng_a.random() == rng_b.random()
    assert all(a.strategy == "uniform_random" for a in agents_a)
    assert all(a.strategy == "moneyline_uniform" for a in agents_b)


def test_mixed_strategy_zero_fraction_is_all_cell():
    agents = _mk_agents(50)
    assign_strategies(agents, "mixed", np.random.default_rng(0), meta_agent_fraction=0.0)
    assert all(a.strategy == "uniform_random" for a in agents)


def test_mixed_strategy_full_fraction_is_all_meta():
    agents = _mk_agents(50)
    assign_strategies(agents, "mixed", np.random.default_rng(0), meta_agent_fraction=1.0)
    assert all(a.strategy == "moneyline_uniform" for a in agents)


def test_mixed_trial_places_both_cell_and_meta_actions():
    """End-to-end: a mixed trial should produce both cell mints and meta-trade fills."""
    cfg = SimConfig(
        n_agents=40,
        balance_min=100.0,
        balance_max=200.0,
        init_mcap_min=0.10,
        init_mcap_max=10.0,
        winner_mode="uniform",
        strategy="mixed",
        meta_agent_fraction=0.8,
        min_mint=1.0,
        max_per_mint=20.0,
        seed=2026,
    )
    rng = np.random.default_rng(cfg.seed)
    trial = run_one_trial(cfg, trial_id=0, rng=rng, log_events=True)

    assert trial.meta_trade_log, "expected some meta-trade fills"
    assert trial.event_log, "expected some events"
    kinds = {e["kind"] for e in trial.event_log}
    assert "meta" in kinds
    assert "cell" in kinds


def test_mixed_trial_conservation():
    """Mixed cohort must still conserve cash: sum(in) == sum(out)."""
    cfg = SimConfig(
        n_agents=30,
        balance_min=50.0,
        balance_max=500.0,
        winner_mode="uniform",
        strategy="mixed",
        meta_agent_fraction=0.8,
        seed=99,
    )
    rng = np.random.default_rng(cfg.seed)
    trial = run_one_trial(cfg, trial_id=0, rng=rng)
    total_in = sum(a.starting_balance for a in trial.agents) + trial.house_seed_total
    total_out = sum(a.terminal_cash for a in trial.agents) + trial.house_terminal_value
    assert math.isclose(total_in, total_out, rel_tol=1e-9, abs_tol=1e-6)


def test_mixed_meta_disabled_rejected():
    with pytest.raises(ValueError):
        SimConfig(strategy="mixed", meta_trades_enabled=False)


def test_mixed_fraction_out_of_range_rejected():
    with pytest.raises(ValueError):
        SimConfig(strategy="mixed", meta_agent_fraction=-0.1)
    with pytest.raises(ValueError):
        SimConfig(strategy="mixed", meta_agent_fraction=1.1)


def test_mixed_assignment_is_seed_reproducible():
    cfg = SimConfig(
        n_agents=50, balance_min=100.0, balance_max=200.0, strategy="mixed", seed=7
    )
    rng1 = np.random.default_rng(cfg.seed)
    rng2 = np.random.default_rng(cfg.seed)
    t1 = run_one_trial(cfg, trial_id=0, rng=rng1)
    t2 = run_one_trial(cfg, trial_id=0, rng=rng2)
    strategies_1 = [a.strategy for a in t1.agents]
    strategies_2 = [a.strategy for a in t2.agents]
    assert strategies_1 == strategies_2
