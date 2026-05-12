"""argparse entry point. Runs the Monte Carlo and writes all artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import viz
from .analytics import (
    build_agent_pnl_df,
    build_event_log_df,
    build_terminal_grids_df,
    build_trials_df,
    conservation_check,
    percentiles,
    winner_frequency_grid,
)
from .market import GRID_SIZE
from .settlement import winner_probabilities
from .simulation import SimConfig, run_monte_carlo


def parse_args(argv=None) -> SimConfig:
    p = argparse.ArgumentParser(description="Dynamic parimutuel Monte Carlo simulator")
    p.add_argument("--n-agents", type=int, default=100)
    p.add_argument("--balance-min", type=float, default=50.0)
    p.add_argument("--balance-max", type=float, default=5000.0)
    p.add_argument("--n-trials", "-S", type=int, default=1000)
    p.add_argument("--init-mcap-min", type=float, default=0.10)
    p.add_argument("--init-mcap-max", type=float, default=10.0)
    p.add_argument(
        "--winner-distribution",
        type=str,
        default="realistic",
        help="uniform | realistic | fixed:i,j",
    )
    p.add_argument(
        "--agent-strategy",
        type=str,
        default="uniform_random",
        choices=["uniform_random", "weighted_by_marginal_payout"],
    )
    p.add_argument("--min-mint", type=float, default=1.0)
    p.add_argument("--max-per-mint", type=float, default=50.0)
    p.add_argument("--min-mint-threshold", type=float, default=0.01)
    p.add_argument("--refund-on-empty-winner", action="store_true", default=True)
    p.add_argument("--no-refund-on-empty-winner", dest="refund_on_empty_winner", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-events-for-first-k", type=int, default=10)
    p.add_argument("--out-root", type=str, default="outputs/runs")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = SimConfig(
        n_agents=args.n_agents,
        balance_min=args.balance_min,
        balance_max=args.balance_max,
        n_trials=args.n_trials,
        init_mcap_min=args.init_mcap_min,
        init_mcap_max=args.init_mcap_max,
        winner_mode=args.winner_distribution,
        strategy=args.agent_strategy,
        min_mint=args.min_mint,
        max_per_mint=args.max_per_mint,
        min_mint_threshold=args.min_mint_threshold,
        refund_on_empty=args.refund_on_empty_winner,
        seed=args.seed,
        log_events_for_first_k=args.log_events_for_first_k,
    )
    return cfg, args


def _echo_config(cfg: SimConfig) -> None:
    print("Resolved configuration:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print()


def main(argv=None) -> int:
    cfg, args = parse_args(argv)
    if not args.quiet:
        _echo_config(cfg)

    out_root = Path(args.out_root)
    run_dir = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing artifacts to: {run_dir}")

    t0 = time.time()
    trials = list(run_monte_carlo(cfg, verbose=not args.quiet))
    elapsed = time.time() - t0
    print(f"\nFinished {cfg.n_trials} trials in {elapsed:.2f}s")

    # --- DataFrames ---
    print("Building dataframes...")
    trials_df = build_trials_df(trials)
    agent_df = build_agent_pnl_df(trials)
    terminal_df = build_terminal_grids_df(trials)
    event_df = build_event_log_df(trials)

    trials_df.to_parquet(run_dir / "trials.parquet")
    agent_df.to_parquet(run_dir / "agent_pnl.parquet")
    terminal_df.to_parquet(run_dir / "terminal_grids.parquet")
    if not event_df.empty:
        event_df.to_parquet(run_dir / "event_log_sample.parquet")
    trials_df.to_csv(run_dir / "trials.csv", index=False)
    agent_df.to_csv(run_dir / "agent_pnl.csv", index=False)

    # --- summary.json ---
    cons = conservation_check(trials)
    expected_winner = winner_probabilities(cfg.winner_mode)
    empirical_winner = winner_frequency_grid(trials_df)
    by_decile = {}
    if len(agent_df) >= 10:
        deciles = pd.qcut(agent_df["starting_balance"], 10, labels=False, duplicates="drop")
        for d in sorted(deciles.dropna().unique()):
            sub = agent_df[deciles == d]
            by_decile[f"decile_{int(d)+1}"] = percentiles(sub["pnl"])
    summary = {
        "params": cfg.__dict__,
        "n_trials": cfg.n_trials,
        "elapsed_seconds": elapsed,
        "pnl_percentiles_overall": percentiles(agent_df["pnl"]),
        "roi_percentiles_overall": percentiles(agent_df["roi"]),
        "pnl_percentiles_by_balance_decile": by_decile,
        "win_rate_median_agent_positive": float(
            (
                agent_df.groupby("trial_id")["pnl"].median() > 0
            ).mean()
        ),
        "mean_gini_supply": float(trials_df["gini_supply"].mean()),
        "mean_gini_terminal_cash": float(trials_df["gini_terminal_cash"].mean()),
        "winner_frequency_grid": empirical_winner.tolist(),
        "expected_winner_grid": expected_winner.tolist(),
        "winner_total_variation_distance": float(
            0.5 * np.abs(empirical_winner - expected_winner).sum()
        ),
        "house_economics": {
            "mean_house_seed_total": float(trials_df["house_seed_total"].mean()),
            "mean_house_pnl": float(trials_df["house_pnl"].mean()),
            "mean_seed_to_pool_ratio": float(
                (trials_df["house_seed_total"] / trials_df["total_pool"].clip(lower=1e-12)).mean()
            ),
        },
        "conservation": cons,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- Validation ---
    nan_inf = {
        col: int(agent_df[col].isna().sum() + np.isinf(agent_df[col]).sum())
        for col in ["starting_balance", "terminal_cash", "pnl", "roi"]
    }
    if cons["max_relative_error"] > 1e-6:
        print(f"  WARNING: max conservation error = {cons['max_relative_error']:.3e}")
    if any(v > 0 for v in nan_inf.values()):
        print(f"  WARNING: found NaN/Inf values: {nan_inf}")

    # --- Plots ---
    print("Rendering plots...")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    viz.plot_mean_terminal_supply(terminal_df, plots_dir / "mean_supply.png")
    viz.plot_mean_marginal_payout(terminal_df, plots_dir / "mean_payout.png")
    viz.plot_realized_payout_when_winner(terminal_df, trials_df, plots_dir / "realized_payout.png")
    viz.plot_pnl_histogram(agent_df, plots_dir / "pnl_distribution.png")
    viz.plot_roi_vs_balance(agent_df, plots_dir / "roi_vs_balance.png")
    viz.plot_lorenz(agent_df, plots_dir / "lorenz.png")
    viz.plot_winner_vs_supply(terminal_df, trials_df, plots_dir / "winner_vs_supply.png")

    print("Building interactive dashboard...")
    viz.build_dashboard(terminal_df, trials_df, agent_df, run_dir / "dashboard.html")

    # --- REPORT.md ---
    print("Writing REPORT.md...")
    write_report(run_dir, cfg, summary, trials_df, agent_df, elapsed)

    print(f"\nDone. Outputs at: {run_dir}")
    return 0


def write_report(run_dir, cfg, summary, trials_df, agent_df, elapsed):
    median_roi = agent_df["roi"].median()
    mean_pnl = agent_df["pnl"].mean()
    pct_pos = (agent_df["pnl"] > 0).mean()
    winner_freq = np.array(summary["winner_frequency_grid"])
    flat_idx = int(np.argmax(winner_freq))
    modal_i, modal_j = divmod(flat_idx, GRID_SIZE)
    modal_label = f"({modal_i}, {modal_j})"

    # Cohort analysis: did agents who concentrated holdings in the modal cell do better?
    # Use the actual winning trials to measure.
    winners_in_modal = trials_df[
        (trials_df.winner_i == modal_i) & (trials_df.winner_j == modal_j)
    ]
    if not winners_in_modal.empty:
        modal_payout = winners_in_modal["payout_per_unit_W"].mean()
    else:
        modal_payout = float("nan")

    tv_dist = summary["winner_total_variation_distance"]
    seed_pool = summary["house_economics"]["mean_seed_to_pool_ratio"]
    mean_pool = trials_df["total_pool"].mean()
    mean_house_pnl = summary["house_economics"]["mean_house_pnl"]
    house_take_pct = mean_house_pnl / mean_pool if mean_pool > 0 else 0.0

    md = f"""# Dynamic Parimutuel Monte Carlo — Report

**Run timestamp:** {run_dir.name}
**Seed:** {cfg.seed}
**Trials:** {cfg.n_trials} · **Agents per trial:** {cfg.n_agents} · **Strategy:** {cfg.strategy}
**Winner prior:** {cfg.winner_mode} · **Wall time:** {elapsed:.1f}s

## Headline stats

| metric | value |
|---|---|
| Total pool (mean across trials) | ${trials_df["total_pool"].mean():,.2f} |
| Payout per unit on winner (mean) | ${trials_df["payout_per_unit_W"].mean():,.2f} |
| Mean agent P&L | ${mean_pnl:,.2f} |
| Median agent ROI | {median_roi:.2%} |
| Agents finishing positive | {pct_pos:.1%} |
| Mean Gini of terminal supply | {summary['mean_gini_supply']:.3f} |
| Mean Gini of terminal cash | {summary['mean_gini_terminal_cash']:.3f} |
| Mean house seed | ${summary['house_economics']['mean_house_seed_total']:,.2f} |
| Mean house P&L | ${summary['house_economics']['mean_house_pnl']:,.2f} |
| Seed / pool ratio | {seed_pool:.4f} |
| Winner empirical vs expected (TV dist) | {tv_dist:.4f} |
| Max conservation error (relative) | {summary['conservation']['max_relative_error']:.2e} |

## Top takeaways

1. **Modal scoreline is {modal_label}.** When it wins, payout-per-unit averages
   **${modal_payout:,.2f}**. The empirical winner distribution matches the
   configured prior with total-variation distance {tv_dist:.4f}.
2. **House take is ~{house_take_pct:.1%} of pool per trial.** The protocol seeds
   only {seed_pool:.2%} of the pool but earns mean P&L of ${mean_house_pnl:,.2f}
   per trial — the seed units in the winning cell get a pro-rata share of the
   *entire* pool, which is much larger than the seed itself. Agents collectively
   lose this same amount (conservation).
3. **Agent fortunes are bimodal:** {pct_pos:.1%} of agents finish positive while
   the rest lose their stake — only holders of the winning cell are paid. Median
   ROI of {median_roi:.2%} reflects this winner-take-all geometry.

## Static plots

- ![](plots/mean_supply.png)
- ![](plots/mean_payout.png)
- ![](plots/realized_payout.png)
- ![](plots/pnl_distribution.png)
- ![](plots/roi_vs_balance.png)
- ![](plots/lorenz.png)
- ![](plots/winner_vs_supply.png)

## Interactive dashboard

Open [`dashboard.html`](dashboard.html) in a browser for KPIs, heatmaps, and a
per-trial slider over the 8×8 final-supply grid.

## Files

- `summary.json` — run parameters and aggregate stats
- `trials.parquet`, `agent_pnl.parquet`, `terminal_grids.parquet` — tabular data
- `event_log_sample.parquet` — full mint log for the first {cfg.log_events_for_first_k} trials
"""
    (run_dir / "REPORT.md").write_text(md)


if __name__ == "__main__":
    sys.exit(main())
