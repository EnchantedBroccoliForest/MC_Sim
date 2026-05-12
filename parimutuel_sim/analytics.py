"""Aggregate statistics over Monte Carlo trial results."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .market import GRID_SIZE, mcap


def gini(x: np.ndarray) -> float:
    """Standard Gini coefficient. Returns 0 for all-equal arrays."""
    x = np.asarray(x, dtype=float).flatten()
    if x.size == 0:
        return 0.0
    if np.all(x <= 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def lorenz_curve(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (population_fraction, cumulative_share_fraction)."""
    x = np.sort(np.asarray(x, dtype=float).flatten())
    n = x.size
    if n == 0 or x.sum() <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    pop = np.arange(1, n + 1) / n
    share = np.cumsum(x) / x.sum()
    pop = np.concatenate([[0.0], pop])
    share = np.concatenate([[0.0], share])
    return pop, share


def build_agent_pnl_df(trials: Iterable) -> pd.DataFrame:
    """One row per (trial_id, agent_id)."""
    rows = []
    for t in trials:
        for a in t.agents:
            rows.append(
                {
                    "trial_id": t.trial_id,
                    "agent_id": a.agent_id,
                    "starting_balance": a.starting_balance,
                    "terminal_cash": a.terminal_cash,
                    "pnl": a.terminal_cash - a.starting_balance,
                    "roi": (a.terminal_cash - a.starting_balance) / a.starting_balance
                    if a.starting_balance > 0
                    else 0.0,
                    "cells_held": a.cells_held,
                    "held_winner_units": a.holdings.get(t.winner, 0.0),
                }
            )
    return pd.DataFrame(rows)


def build_trials_df(trials: Iterable) -> pd.DataFrame:
    rows = []
    for t in trials:
        n_profit = sum(1 for a in t.agents if a.terminal_cash > a.starting_balance)
        rows.append(
            {
                "trial_id": t.trial_id,
                "winner_i": t.winner[0],
                "winner_j": t.winner[1],
                "total_pool": t.total_pool,
                "payout_per_unit_W": t.payout_per_unit,
                "n_agents_profitable": n_profit,
                "n_agents": len(t.agents),
                "house_seed_total": t.house_seed_total,
                "house_terminal_value": t.house_terminal_value,
                "house_pnl": t.house_terminal_value - t.house_seed_total,
                "gini_supply": gini(t.final_supply.flatten()),
                "gini_terminal_cash": gini(
                    np.array([a.terminal_cash for a in t.agents])
                ),
            }
        )
    return pd.DataFrame(rows)


def build_terminal_grids_df(trials: Iterable) -> pd.DataFrame:
    """One row per (trial, cell) with final supply/mcap/price/payout/holders."""
    rows = []
    for t in trials:
        cap = mcap(t.final_supply)
        price = np.power(t.final_supply, 3.0 / 4.0)
        payout = (4.0 / 7.0) * t.final_supply
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rows.append(
                    {
                        "trial_id": t.trial_id,
                        "i": i,
                        "j": j,
                        "supply": float(t.final_supply[i, j]),
                        "mcap": float(cap[i, j]),
                        "marginal_price": float(price[i, j]),
                        "marginal_payout": float(payout[i, j]),
                        "n_holders": int(t.n_holders[i, j]),
                        "is_winner": (i, j) == t.winner,
                    }
                )
    return pd.DataFrame(rows)


def build_event_log_df(trials: Iterable) -> pd.DataFrame:
    rows: List[dict] = []
    for t in trials:
        if t.event_log:
            rows.extend(t.event_log)
    return pd.DataFrame(rows)


def percentiles(s: pd.Series, pcts=(1, 5, 25, 50, 75, 95, 99)) -> dict:
    if len(s) == 0:
        return {f"p{p}": float("nan") for p in pcts}
    arr = s.to_numpy(dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in pcts}


def winner_frequency_grid(trials_df: pd.DataFrame) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    for _, row in trials_df.iterrows():
        grid[int(row.winner_i), int(row.winner_j)] += 1.0
    if grid.sum() > 0:
        grid /= grid.sum()
    return grid


def mean_terminal_supply_grid(terminal_df: pd.DataFrame) -> np.ndarray:
    g = (
        terminal_df.groupby(["i", "j"])["supply"]
        .mean()
        .reindex(
            pd.MultiIndex.from_product([range(GRID_SIZE), range(GRID_SIZE)], names=["i", "j"]),
            fill_value=0,
        )
        .to_numpy()
        .reshape(GRID_SIZE, GRID_SIZE)
    )
    return g


def mean_marginal_payout_grid(terminal_df: pd.DataFrame) -> np.ndarray:
    return (4.0 / 7.0) * mean_terminal_supply_grid(terminal_df)


def mean_payout_per_unit_when_winner(terminal_df: pd.DataFrame, trials_df: pd.DataFrame) -> np.ndarray:
    """For each cell, average payout_per_unit across trials where that cell won."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for _, row in trials_df.iterrows():
        i, j = int(row.winner_i), int(row.winner_j)
        grid[i, j] += float(row.payout_per_unit_W)
        counts[i, j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, grid / np.maximum(counts, 1), np.nan)
    return out


def conservation_check(trials) -> dict:
    """Sanity-check that total in == total out within tolerance, per trial."""
    max_err_rel = 0.0
    n = 0
    for t in trials:
        n += 1
        total_in = sum(a.starting_balance for a in t.agents) + t.house_seed_total
        total_out = (
            sum(a.terminal_cash for a in t.agents) + t.house_terminal_value
        )
        if total_in > 0:
            err = abs(total_in - total_out) / total_in
            max_err_rel = max(max_err_rel, err)
    return {"max_relative_error": max_err_rel, "n_trials_checked": n}
