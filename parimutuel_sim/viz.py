"""Static plots and interactive HTML dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from .analytics import (
    lorenz_curve,
    mean_marginal_payout_grid,
    mean_payout_per_unit_when_winner,
    mean_terminal_supply_grid,
    winner_frequency_grid,
)
from .market import GRID_SIZE
from .meta_trades import META_TRADES

ROW_LABELS = ["0", "1", "2", "3", "4", "5", "6", "≥7"]
COL_LABELS = ["0", "1", "2", "3", "4", "5", "6", "≥7"]


def _heatmap(ax, grid, title, cmap="viridis", fmt=".2f", log_color=False):
    if log_color:
        norm = matplotlib.colors.LogNorm(
            vmin=max(np.nanmin(grid[grid > 0]) if np.any(grid > 0) else 1e-6, 1e-6),
            vmax=max(np.nanmax(grid), 1e-6),
        )
        sns.heatmap(
            grid, ax=ax, cmap=cmap, norm=norm,
            xticklabels=COL_LABELS, yticklabels=ROW_LABELS,
            cbar_kws={"label": "log scale"}, annot=False,
        )
    else:
        sns.heatmap(
            grid, ax=ax, cmap=cmap, annot=True, fmt=fmt,
            xticklabels=COL_LABELS, yticklabels=ROW_LABELS,
        )
    ax.set_xlabel("CRY goals")
    ax.set_ylabel("MCI goals")
    ax.set_title(title)


def plot_mean_terminal_supply(terminal_df: pd.DataFrame, out: Path) -> None:
    grid = mean_terminal_supply_grid(terminal_df)
    fig, ax = plt.subplots(figsize=(7, 6))
    _heatmap(ax, grid, "Mean terminal OT supply (log color)", log_color=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_mean_marginal_payout(terminal_df: pd.DataFrame, out: Path) -> None:
    """Mean marginal_payout heatmap; reverse colour so higher payout looks 'hotter'."""
    grid = mean_marginal_payout_grid(terminal_df)
    fig, ax = plt.subplots(figsize=(7, 6))
    _heatmap(ax, grid, "Mean marginal payout multiplier", cmap="magma_r")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_realized_payout_when_winner(
    terminal_df: pd.DataFrame, trials_df: pd.DataFrame, out: Path
) -> None:
    grid = mean_payout_per_unit_when_winner(terminal_df, trials_df)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        grid, ax=ax, cmap="rocket_r", annot=True, fmt=".1f",
        xticklabels=COL_LABELS, yticklabels=ROW_LABELS,
        cbar_kws={"label": "$ per unit"},
    )
    ax.set_xlabel("CRY goals")
    ax.set_ylabel("MCI goals")
    ax.set_title("Realized payout per unit (avg over trials where cell won)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_pnl_histogram(agent_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if agent_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No agent data", ha="center", va="center")
            ax.set_axis_off()
    else:
        sns.histplot(agent_df["pnl"], bins=80, ax=axes[0], color="steelblue")
        axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[0].set_title("Agent P&L distribution (overall)")
        axes[0].set_xlabel("P&L ($)")

        n_balance_levels = agent_df["starting_balance"].nunique(dropna=True)
        if n_balance_levels > 1:
            q = min(4, n_balance_levels)
            quartiles = pd.qcut(
                agent_df["starting_balance"], q, labels=False, duplicates="drop"
            )
            labels = quartiles.map(lambda q: f"Q{int(q) + 1}" if pd.notna(q) else "NA")
            df = agent_df.assign(balance_q=labels)
        else:
            df = agent_df.assign(balance_q="All")
        sns.violinplot(data=df, x="balance_q", y="roi", ax=axes[1], cut=0)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("ROI by starting-balance cohort")
        axes[1].set_xlabel("Starting balance cohort")
        axes[1].set_ylabel("ROI")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_roi_vs_balance(agent_df: pd.DataFrame, out: Path) -> None:
    df = agent_df.sample(min(20000, len(agent_df)), random_state=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    if df.empty:
        ax.text(0.5, 0.5, "No agent data", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.scatter(df["starting_balance"], df["roi"], s=4, alpha=0.2)
        if df["starting_balance"].nunique(dropna=True) > 1:
            # Simple binned mean (LOESS-style without scipy.loess)
            bins = np.linspace(df["starting_balance"].min(), df["starting_balance"].max(), 25)
            df2 = df.assign(bin=pd.cut(df["starting_balance"], bins, duplicates="drop"))
            binned = (
                df2.groupby("bin", observed=True)["roi"]
                .agg(["mean", "count"])
                .reset_index()
            )
            centers = [(b.left + b.right) / 2 for b in binned["bin"]]
            ax.plot(
                centers,
                binned["mean"],
                color="crimson",
                linewidth=2,
                label="binned mean ROI",
            )
            ax.legend()
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Starting balance ($)")
        ax.set_ylabel("ROI")
        ax.set_title("ROI vs. starting balance")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_lorenz(agent_df: pd.DataFrame, out: Path) -> None:
    pop, share = lorenz_curve(agent_df["terminal_cash"].to_numpy())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pop, share, color="steelblue", linewidth=2, label="terminal cash")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="equality")
    ax.fill_between(pop, share, pop, alpha=0.15, color="steelblue")
    ax.set_xlabel("Cumulative share of agents")
    ax.set_ylabel("Cumulative share of terminal cash")
    ax.set_title("Lorenz curve — agent terminal cash")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_winner_vs_supply(
    terminal_df: pd.DataFrame, trials_df: pd.DataFrame, out: Path
) -> None:
    win_freq = winner_frequency_grid(trials_df).flatten()
    mean_supply = mean_terminal_supply_grid(terminal_df).flatten()
    if mean_supply.sum() > 0:
        supply_share = mean_supply / mean_supply.sum()
    else:
        supply_share = mean_supply
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(supply_share, win_freq, s=30, alpha=0.7)
    lim = max(supply_share.max(), win_freq.max()) * 1.1 + 1e-9
    ax.plot([0, lim], [0, lim], "--", color="grey", label="x=y")
    ax.set_xlabel("Mean supply share of cell")
    ax.set_ylabel("Empirical win frequency")
    ax.set_title("Where agents put capital vs. where wins land")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _meta_trades_section(
    meta_legs_df: Optional[pd.DataFrame],
    meta_summary_df: Optional[pd.DataFrame],
    meta_share_grid: Optional[np.ndarray],
) -> str:
    """Render the Meta Trades dashboard tab. Empty string if no meta data."""
    if meta_legs_df is None or meta_legs_df.empty:
        return ""

    summary = meta_summary_df if meta_summary_df is not None else pd.DataFrame()

    # Bar: total cash per meta key.
    if not summary.empty:
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=summary["meta_key"],
                    y=summary["total_cash"],
                    marker_color=["#4C78A8", "#54A24B", "#E45756"][: len(summary)],
                    text=[f"${v:,.0f}" for v in summary["total_cash"]],
                    textposition="outside",
                )
            ]
        ).update_layout(
            title="Total cash routed per meta key",
            height=380,
            xaxis_title="Meta key",
            yaxis_title="Cash ($)",
        )
        bar_html = fig_bar.to_html(full_html=False, include_plotlyjs=False)
    else:
        bar_html = ""

    # Stacked time-series across ticks: cash per tick into each meta key.
    if "tick" in meta_legs_df.columns and meta_legs_df["tick"].notna().any():
        per_tick = (
            meta_legs_df.dropna(subset=["tick"])
            .groupby(["tick", "meta_key"], as_index=False)["cash"]
            .sum()
        )
        fig_ts = go.Figure()
        palette = {"MCI_WIN": "#4C78A8", "DRAW": "#54A24B", "CRY_WIN": "#E45756"}
        for key in ["MCI_WIN", "DRAW", "CRY_WIN"]:
            sub = per_tick[per_tick.meta_key == key].sort_values("tick")
            fig_ts.add_trace(
                go.Bar(
                    x=sub["tick"],
                    y=sub["cash"],
                    name=key,
                    marker_color=palette.get(key, None),
                )
            )
        fig_ts.update_layout(
            barmode="stack",
            title="Cash routed per tick by meta key",
            height=380,
            xaxis_title="Tick (within trial)",
            yaxis_title="Cash ($)",
        )
        ts_html = fig_ts.to_html(full_html=False, include_plotlyjs=False)
    else:
        ts_html = ""

    # Per-bucket heatmaps: share of final mcap arriving via meta.
    hm_blocks = []
    if meta_share_grid is not None:
        for key, mdef in META_TRADES.items():
            mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
            for (i, j) in mdef.cells:
                mask[i, j] = float(meta_share_grid[i, j])
            fig = go.Figure(
                data=go.Heatmap(
                    z=mask,
                    x=COL_LABELS,
                    y=ROW_LABELS,
                    colorscale="Viridis",
                    zmin=0.0,
                    zmax=1.0,
                    hovertemplate=(
                        "MCI %{y} − CRY %{x}<br>meta share: %{z:.1%}<extra></extra>"
                    ),
                )
            ).update_layout(
                title=f"{mdef.display_name}: meta share of cell mcap delta",
                height=380,
                xaxis_title="CRY goals",
                yaxis_title="MCI goals",
            )
            hm_blocks.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return (
        "<h2>Meta Trades</h2>"
        + bar_html
        + ts_html
        + "".join(hm_blocks)
    )


def build_dashboard(
    terminal_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    out: Path,
    meta_legs_df: Optional[pd.DataFrame] = None,
    meta_summary_df: Optional[pd.DataFrame] = None,
    meta_share_grid: Optional[np.ndarray] = None,
) -> None:
    """Single interactive HTML report with KPIs + heatmaps + slider."""

    mean_supply = mean_terminal_supply_grid(terminal_df)
    mean_payout = mean_marginal_payout_grid(terminal_df)
    realized = mean_payout_per_unit_when_winner(terminal_df, trials_df)
    win_freq = winner_frequency_grid(trials_df)

    kpi_html = f"""
    <h2>Run summary</h2>
    <ul>
      <li>Trials: <b>{len(trials_df)}</b></li>
      <li>Agents per trial: <b>{int(trials_df['n_agents'].mean())}</b></li>
      <li>Mean total pool: <b>${trials_df['total_pool'].mean():,.2f}</b></li>
      <li>Mean payout-per-unit (winner): <b>${trials_df['payout_per_unit_W'].mean():,.2f}</b></li>
      <li>Mean agent P&L: <b>${agent_df['pnl'].mean():,.2f}</b></li>
      <li>Median agent ROI: <b>{agent_df['roi'].median():.2%}</b></li>
      <li>% agents finishing positive: <b>{(agent_df['pnl'] > 0).mean():.1%}</b></li>
      <li>Mean house P&amp;L: <b>${trials_df['house_pnl'].mean():,.2f}</b></li>
    </ul>
    """

    def hm(z, title, colorscale="Viridis"):
        return go.Figure(
            data=go.Heatmap(
                z=z, x=COL_LABELS, y=ROW_LABELS, colorscale=colorscale,
                hovertemplate="MCI %{y} − CRY %{x}<br>value: %{z:.3f}<extra></extra>",
            )
        ).update_layout(title=title, height=400, xaxis_title="CRY goals", yaxis_title="MCI goals")

    fig_supply = hm(mean_supply, "Mean terminal supply")
    fig_payout = hm(mean_payout, "Mean marginal payout multiplier", "Magma")
    fig_realized = hm(realized, "Realized payout / unit (when cell wins)", "Reds")
    fig_winfreq = hm(win_freq, "Empirical winner frequency", "Blues")

    # Trial slider: show final supply heatmap per trial for first 50 trials
    slider_trials = sorted(terminal_df["trial_id"].unique())[:50]
    frames = []
    for tid in slider_trials:
        sub = (
            terminal_df[terminal_df.trial_id == tid]
            .pivot(index="i", columns="j", values="supply")
            .reindex(index=range(GRID_SIZE), columns=range(GRID_SIZE))
            .to_numpy()
        )
        frames.append(go.Frame(data=[go.Heatmap(z=sub, x=COL_LABELS, y=ROW_LABELS, colorscale="Viridis")], name=str(tid)))
    if frames:
        init = frames[0].data[0].z
    else:
        init = mean_supply
    fig_slider = go.Figure(
        data=[go.Heatmap(z=init, x=COL_LABELS, y=ROW_LABELS, colorscale="Viridis")],
        frames=frames,
    )
    fig_slider.update_layout(
        title="Per-trial final supply (first 50 trials)",
        height=450,
        xaxis_title="CRY goals",
        yaxis_title="MCI goals",
        sliders=[
            {
                "steps": [
                    {"args": [[f.name], {"frame": {"duration": 0, "redraw": True}}], "label": f.name, "method": "animate"}
                    for f in frames
                ],
                "currentvalue": {"prefix": "trial_id="},
            }
        ] if frames else [],
    )

    # PnL histogram
    fig_hist = px.histogram(
        agent_df, x="pnl", nbins=80, title="Agent P&L distribution",
    ).update_layout(height=350)

    meta_html = _meta_trades_section(meta_legs_df, meta_summary_df, meta_share_grid)

    parts = [
        "<html><head><title>Parimutuel MC Report</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:20px auto;padding:0 16px}h2{margin-top:32px}</style>",
        "</head><body>",
        "<h1>Dynamic Parimutuel Monte Carlo</h1>",
        kpi_html,
        "<h2>Grid heatmaps</h2>",
        fig_supply.to_html(full_html=False, include_plotlyjs="cdn"),
        fig_payout.to_html(full_html=False, include_plotlyjs=False),
        fig_realized.to_html(full_html=False, include_plotlyjs=False),
        fig_winfreq.to_html(full_html=False, include_plotlyjs=False),
        "<h2>P&L distribution</h2>",
        fig_hist.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Per-trial supply</h2>",
        fig_slider.to_html(full_html=False, include_plotlyjs=False),
        meta_html,
        "</body></html>",
    ]
    out.write_text("\n".join(parts))
