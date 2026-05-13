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


def score_label(i: int, j: int) -> str:
    """Human-readable scoreline for grid cell (i, j). MCI on the left, CRY on the right."""
    a = "7+" if i >= GRID_SIZE - 1 else str(i)
    b = "7+" if j >= GRID_SIZE - 1 else str(j)
    return f"{a}-{b}"


def score_label_long(i: int, j: int) -> str:
    return f"MCI {score_label(i, j)} CRY"


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


def plot_capital_vs_wins(
    terminal_df: pd.DataFrame, trials_df: pd.DataFrame, out: Path
) -> None:
    """Two-panel heatmap: where capital landed vs. where wins landed.

    Cells are annotated with the human-readable scoreline (e.g. "2-1"), and the
    modal winning cell is outlined so the eye lands on it immediately. This
    single chart replaces the separate supply / win-frequency / scatter views.
    """
    mean_supply = mean_terminal_supply_grid(terminal_df)
    supply_share = (
        mean_supply / mean_supply.sum() if mean_supply.sum() > 0 else mean_supply
    )
    win_freq = winner_frequency_grid(trials_df)

    annot = np.array(
        [[score_label(i, j) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, grid, title, cmap in [
        (axes[0], supply_share, "Where capital went (share of mean supply)", "Blues"),
        (axes[1], win_freq, "Where wins landed (empirical frequency)", "Greens"),
    ]:
        sns.heatmap(
            grid,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt="",
            xticklabels=COL_LABELS,
            yticklabels=ROW_LABELS,
            cbar_kws={"label": "share"},
            annot_kws={"fontsize": 8, "color": "#333"},
        )
        ax.set_xlabel("CRY goals")
        ax.set_ylabel("MCI goals")
        ax.set_title(title)
        # Outline the modal cell of this panel.
        flat = int(np.argmax(grid))
        mi, mj = divmod(flat, GRID_SIZE)
        ax.add_patch(
            plt.Rectangle((mj, mi), 1, 1, fill=False, edgecolor="crimson", lw=2)
        )

    fig.suptitle("Capital allocation vs. realised wins (red box = modal cell)")
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
    if not summary.empty:
        summary = summary[summary["n_trades"] > 0].copy()

    # Bar: total cash per meta key (only keys that were actually traded).
    if not summary.empty:
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=summary["meta_key"],
                    y=summary["total_cash"],
                    marker_color="#2f6feb",
                    text=[f"${v:,.0f}" for v in summary["total_cash"]],
                    textposition="outside",
                )
            ]
        ).update_layout(
            height=320,
            margin=dict(t=20, l=40, r=20, b=60),
            xaxis_title="Meta key",
            yaxis_title="Cash routed ($)",
        )
        bar_html = fig_bar.to_html(full_html=False, include_plotlyjs=False)
    else:
        bar_html = ""

    # Per-bucket heatmaps: only for meta keys that actually saw trades. One
    # combined heatmap-per-set keeps the report short instead of dumping every
    # bucket regardless of whether it was used.
    hm_blocks = []
    if meta_share_grid is not None and not summary.empty:
        active_keys = set(summary["meta_key"].tolist())
        for key in active_keys:
            mdef = META_TRADES.get(key)
            if mdef is None:
                continue
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
                height=340,
                xaxis_title="CRY goals",
                yaxis_title="MCI goals",
            )
            hm_blocks.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if not bar_html and not hm_blocks:
        return ""

    return (
        "<h2>Meta trades</h2>"
        "<p class='caption'>Cash routed via named buckets (moneyline, spreads, totals, exact-score). "
        "Heatmaps below show, for each active bucket, the fraction of each cell's market-cap "
        "growth that arrived via that meta trade.</p>"
        + ("<div class='card'>" + bar_html + "</div>" if bar_html else "")
        + ("<div class='card'>" + "".join(hm_blocks) + "</div>" if hm_blocks else "")
    )


_DASHBOARD_CSS = """
:root {
  --bg: #f7f8fa;
  --card: #ffffff;
  --ink: #1f2430;
  --muted: #5b6472;
  --accent: #2f6feb;
  --good: #2e7d32;
  --bad: #c62828;
  --border: #e3e6ec;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--ink);
  max-width: 1180px;
  margin: 24px auto;
  padding: 0 20px;
  line-height: 1.5;
}
h1 { font-size: 1.7rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 28px 0 6px; }
.subtitle { color: var(--muted); margin: 0 0 22px; }
.headline {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  padding: 14px 18px;
  margin: 12px 0 22px;
  font-size: 0.97rem;
}
.kpis {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin: 4px 0 18px;
}
.kpi {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
}
.kpi .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi .val   { font-size: 1.25rem; font-weight: 600; margin-top: 2px; }
.kpi.good .val { color: var(--good); }
.kpi.bad  .val { color: var(--bad); }
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  margin: 8px 0 18px;
}
.caption { color: var(--muted); font-size: 0.88rem; margin: 2px 4px 8px; }
table.meta { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
table.meta th, table.meta td { padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; }
table.meta th { color: var(--muted); font-weight: 500; background: #fafbfd; }
"""


def _kpi(label: str, value: str, tone: str = "") -> str:
    cls = f"kpi {tone}".strip()
    return f'<div class="{cls}"><div class="label">{label}</div><div class="val">{value}</div></div>'


def build_dashboard(
    terminal_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    out: Path,
    meta_legs_df: Optional[pd.DataFrame] = None,
    meta_summary_df: Optional[pd.DataFrame] = None,
    meta_share_grid: Optional[np.ndarray] = None,
) -> None:
    """Single interactive HTML report. KPI cards, captioned heatmaps, no clutter."""

    mean_supply = mean_terminal_supply_grid(terminal_df)
    supply_share = (
        mean_supply / mean_supply.sum() if mean_supply.sum() > 0 else mean_supply
    )
    realized = mean_payout_per_unit_when_winner(terminal_df, trials_df)
    win_freq = winner_frequency_grid(trials_df)

    # Headline numbers.
    n_trials = len(trials_df)
    mean_pool = float(trials_df["total_pool"].mean())
    mean_payout_W = float(trials_df["payout_per_unit_W"].mean())
    mean_pnl = float(agent_df["pnl"].mean())
    median_roi = float(agent_df["roi"].median())
    pct_pos = float((agent_df["pnl"] > 0).mean())
    mean_house_pnl = float(trials_df["house_pnl"].mean())
    house_take_pct = (mean_house_pnl / mean_pool) if mean_pool > 0 else 0.0

    flat_idx = int(np.argmax(win_freq))
    modal_i, modal_j = divmod(flat_idx, GRID_SIZE)
    modal_freq = float(win_freq[modal_i, modal_j])
    modal_score = score_label_long(modal_i, modal_j)

    headline = (
        f"<b>{n_trials:,} trials.</b> The modal scoreline is "
        f"<b>{modal_score}</b> ({modal_freq:.1%} of trials). "
        f"Agents who hold the winning cell are paid pro-rata; "
        f"<b>{pct_pos:.0%}</b> of agents finish positive, "
        f"the house keeps <b>{house_take_pct:.1%}</b> of the pool on average."
    )

    kpis = [
        _kpi("Trials", f"{n_trials:,}"),
        _kpi("Mean pool", f"${mean_pool:,.0f}"),
        _kpi("Payout / unit (winner)", f"${mean_payout_W:,.2f}"),
        _kpi("Mean agent P&L", f"${mean_pnl:,.2f}", "good" if mean_pnl >= 0 else "bad"),
        _kpi("Median agent ROI", f"{median_roi:.1%}", "good" if median_roi >= 0 else "bad"),
        _kpi("Agents in profit", f"{pct_pos:.0%}"),
        _kpi("Mean house P&L", f"${mean_house_pnl:,.2f}", "good" if mean_house_pnl >= 0 else "bad"),
        _kpi("Modal scoreline", score_label(modal_i, modal_j)),
    ]
    kpi_html = '<div class="kpis">' + "".join(kpis) + "</div>"

    def _grid_subplot(z, colorscale, name, hover_unit):
        annot = [[score_label(i, j) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)]
        return go.Heatmap(
            z=z, x=COL_LABELS, y=ROW_LABELS,
            colorscale=colorscale,
            text=annot, texttemplate="%{text}",
            textfont={"size": 9, "color": "#333"},
            hovertemplate=("MCI %{y} − CRY %{x}<br>" + name + ": %{z:" + hover_unit + "}<extra></extra>"),
            colorbar=dict(thickness=10, len=0.85),
        )

    # 2x2 panel: capital vs wins on top, marginal price vs realised payout on bottom.
    fig_grid = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Capital share (mean supply)",
            "Win frequency",
            "Mean marginal payout multiplier",
            "Realised payout per unit (when cell wins)",
        ),
        horizontal_spacing=0.12, vertical_spacing=0.16,
    )
    fig_grid.add_trace(_grid_subplot(supply_share, "Blues", "share", ".2%"), 1, 1)
    fig_grid.add_trace(_grid_subplot(win_freq, "Greens", "freq", ".2%"), 1, 2)
    fig_grid.add_trace(
        _grid_subplot(mean_marginal_payout_grid(terminal_df), "Magma", "x", ".2f"), 2, 1,
    )
    realized_safe = np.nan_to_num(realized, nan=0.0)
    fig_grid.add_trace(_grid_subplot(realized_safe, "Reds", "$/unit", ".1f"), 2, 2)
    for r in (1, 2):
        for c in (1, 2):
            fig_grid.update_xaxes(title_text="CRY goals" if r == 2 else "", row=r, col=c)
            fig_grid.update_yaxes(title_text="MCI goals" if c == 1 else "", row=r, col=c, autorange="reversed")
    fig_grid.update_layout(height=820, showlegend=False, margin=dict(t=60, l=40, r=20, b=40))

    grid_caption = (
        "Top row tells the calibration story: green ≈ blue means agent capital tracks "
        "real win frequency. Bottom row is the unit economics — the right-hand heatmap "
        "shows what one OT pays out when the cell wins."
    )

    # P&L distribution.
    fig_hist = px.histogram(
        agent_df, x="pnl", nbins=70,
        color_discrete_sequence=["#2f6feb"],
    ).update_layout(
        height=330,
        margin=dict(t=20, l=40, r=20, b=40),
        xaxis_title="Agent P&L ($)",
        yaxis_title="Agents",
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#444")

    hist_caption = (
        "Winner-take-all geometry: a large pile of small losses (agents who held "
        "non-winning cells), plus a long right tail of large wins."
    )

    meta_html = _meta_trades_section(meta_legs_df, meta_summary_df, meta_share_grid)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Parimutuel MC Report</title>",
        f"<style>{_DASHBOARD_CSS}</style>",
        "</head><body>",
        "<h1>Dynamic Parimutuel Monte Carlo — Report</h1>",
        f"<p class='subtitle'>{n_trials:,} trials, {int(trials_df['n_agents'].mean())} agents per trial.</p>",
        f"<div class='headline'>{headline}</div>",
        kpi_html,
        "<h2>Grid story — where capital went vs. where wins landed</h2>",
        f"<p class='caption'>{grid_caption}</p>",
        "<div class='card'>",
        fig_grid.to_html(full_html=False, include_plotlyjs="cdn"),
        "</div>",
        "<h2>Agent outcomes</h2>",
        f"<p class='caption'>{hist_caption}</p>",
        "<div class='card'>",
        fig_hist.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        meta_html,
        "</body></html>",
    ]
    out.write_text("\n".join(parts))
