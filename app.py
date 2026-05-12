"""Flask web UI for the parimutuel Monte Carlo simulator."""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from typing import Iterator

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, Response, render_template, request, stream_with_context, send_file

from parimutuel_sim.analytics import (
    build_agent_pnl_df,
    build_terminal_grids_df,
    build_trials_df,
    conservation_check,
    lorenz_curve,
    mean_marginal_payout_grid,
    mean_payout_per_unit_when_winner,
    mean_terminal_supply_grid,
    winner_frequency_grid,
)
from parimutuel_sim.market import GRID_SIZE
from parimutuel_sim.settlement import winner_probabilities
from parimutuel_sim.simulation import SimConfig, run_monte_carlo

app = Flask(__name__)

ROW_LABELS = ["0", "1", "2", "3", "4", "5", "6", "≥7"]
COL_LABELS = ["0", "1", "2", "3", "4", "5", "6", "≥7"]

# In-memory run store: run_id -> {"status", "progress", "result"}
_runs: dict[str, dict] = {}


def _run_simulation(run_id: str, cfg: SimConfig) -> None:
    store = _runs[run_id]
    store["status"] = "running"
    store["progress"] = 0

    trials = []
    checkpoint = max(1, cfg.n_trials // 20)
    for result in run_monte_carlo(cfg, verbose=False):
        trials.append(result)
        pct = int(len(trials) / cfg.n_trials * 100)
        store["progress"] = pct
        if len(trials) % checkpoint == 0:
            store["progress_message"] = f"Trial {len(trials)} / {cfg.n_trials}"

    # Build analytics
    trials_df = build_trials_df(trials)
    agent_df = build_agent_pnl_df(trials)
    terminal_df = build_terminal_grids_df(trials)
    cons = conservation_check(trials)

    # --- KPIs ---
    mean_pool = float(trials_df["total_pool"].mean())
    mean_ppu = float(trials_df["payout_per_unit_W"].mean())
    mean_pnl = float(agent_df["pnl"].mean())
    median_roi = float(agent_df["roi"].median())
    pct_pos = float((agent_df["pnl"] > 0).mean())
    mean_house_pnl = float(trials_df["house_pnl"].mean())
    mean_gini = float(trials_df["gini_supply"].mean())

    # --- Charts ---
    mean_supply = mean_terminal_supply_grid(terminal_df)
    mean_payout = mean_marginal_payout_grid(terminal_df)
    realized = mean_payout_per_unit_when_winner(terminal_df, trials_df)
    win_freq = winner_frequency_grid(trials_df)

    # Compute extra stats for chart annotations
    gini_overall = float(trials_df["gini_supply"].mean())
    median_pnl = float(agent_df["pnl"].median())
    mean_starting = float(agent_df["starting_balance"].mean())
    n_obs = len(agent_df)

    def titled(title: str, subtitle: str) -> str:
        # Plotly supports HTML in title text. Subtitle = small grey line below.
        return (
            f"<b>{title}</b><br>"
            f"<span style='font-size:11px;color:#94a3b8'>{subtitle}</span>"
        )

    def hm_json(
        z,
        title,
        subtitle,
        colorbar_title,
        hover_value_label,
        colorscale="Viridis",
        fmt=".3f",
        hover_prefix="",
        hover_suffix="",
    ):
        hover = (
            "<b>Score: Man City %{y} — Crystal Palace %{x}</b>"
            f"<br>{hover_value_label}: {hover_prefix}%{{z:{fmt}}}{hover_suffix}"
            "<extra></extra>"
        )
        fig = go.Figure(
            data=go.Heatmap(
                z=z.tolist(),
                x=COL_LABELS,
                y=ROW_LABELS,
                colorscale=colorscale,
                hovertemplate=hover,
                colorbar=dict(
                    title=dict(text=colorbar_title, side="right", font=dict(size=11)),
                    thickness=12,
                    tickfont=dict(size=10),
                ),
            )
        ).update_layout(
            title=dict(text=titled(title, subtitle), x=0.02, xanchor="left"),
            height=410,
            margin=dict(l=70, r=20, t=80, b=70),
            xaxis=dict(
                title=dict(text="Crystal Palace — goals scored", font=dict(size=11)),
                tickmode="array",
                tickvals=COL_LABELS,
                ticktext=COL_LABELS,
                constrain="domain",
            ),
            yaxis=dict(
                title=dict(text="Manchester City — goals scored", font=dict(size=11)),
                tickmode="array",
                tickvals=ROW_LABELS,
                ticktext=ROW_LABELS,
                scaleanchor="x",
                scaleratio=1,
            ),
        )
        return fig.to_json()

    # ── Agent P&L histogram ──
    fig_pnl = px.histogram(
        agent_df, x="pnl", nbins=80,
        color_discrete_sequence=["#4C78A8"],
    ).update_layout(
        title=dict(
            text=titled(
                "Agent profit &amp; loss distribution",
                f"P&amp;L per (agent × trial) — {n_obs:,} observations across {cfg.n_trials} trials",
            ),
            x=0.02, xanchor="left",
        ),
        height=380,
        margin=dict(l=70, r=20, t=80, b=70),
        xaxis_title="Profit / loss per agent ($USD)",
        yaxis_title="Number of (agent × trial) observations",
        bargap=0.02,
    )
    fig_pnl.update_traces(
        hovertemplate="P&L: $%{x:.2f}<br>Count: %{y:,}<extra></extra>"
    )
    fig_pnl.add_vline(
        x=0, line_dash="dash", line_color="#94a3b8", line_width=1,
        annotation_text="break-even", annotation_position="top",
        annotation_font_color="#94a3b8", annotation_font_size=10,
    )
    fig_pnl.add_vline(
        x=median_pnl, line_dash="dot", line_color="#f59e0b", line_width=1.5,
        annotation_text=f"median = ${median_pnl:,.2f}", annotation_position="top right",
        annotation_font_color="#f59e0b", annotation_font_size=10,
    )

    # ── Lorenz curve ──
    pop, share = lorenz_curve(agent_df["terminal_cash"].to_numpy())
    # Gini = 1 - 2 * area under Lorenz curve (trapezoidal estimate).
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    gini_cash = float(1.0 - 2.0 * _trapz(share, pop))
    fig_lorenz = go.Figure()
    fig_lorenz.add_trace(go.Scatter(
        x=pop.tolist(), y=share.tolist(), mode="lines",
        name="Actual distribution",
        line=dict(color="#4C78A8", width=2.5),
        hovertemplate="Poorest %{x:.0%} of agents<br>own %{y:.1%} of total cash<extra></extra>",
        fill="tozeroy", fillcolor="rgba(76, 120, 168, 0.15)",
    ))
    fig_lorenz.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Perfect equality",
        line=dict(color="#94a3b8", dash="dash", width=1.5),
        hovertemplate="Perfect equality<extra></extra>",
    ))
    fig_lorenz.update_layout(
        title=dict(
            text=titled(
                "Inequality of agent terminal cash (Lorenz curve)",
                f"Gini coefficient = {gini_cash:.3f}  ·  0 = perfect equality, 1 = one agent owns everything",
            ),
            x=0.02, xanchor="left",
        ),
        height=380,
        margin=dict(l=70, r=20, t=80, b=70),
        xaxis=dict(title="Cumulative share of agents (poorest → richest)", tickformat=".0%"),
        yaxis=dict(title="Cumulative share of terminal cash", tickformat=".0%"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )

    # ── ROI vs starting balance scatter ──
    sample = agent_df.sample(min(5000, len(agent_df)), random_state=0)
    fig_roi = px.scatter(
        sample, x="starting_balance", y="roi", opacity=0.35,
        color_discrete_sequence=["#4C78A8"],
    ).update_layout(
        title=dict(
            text=titled(
                "Per-agent return on investment vs. starting balance",
                f"{len(sample):,} sampled agents  ·  ROI = (terminal cash − starting cash) ÷ starting cash",
            ),
            x=0.02, xanchor="left",
        ),
        height=380,
        margin=dict(l=70, r=20, t=80, b=70),
        xaxis_title="Starting cash ($USD)",
        yaxis=dict(title="Return on investment (ROI)", tickformat=".0%"),
    )
    fig_roi.update_traces(
        hovertemplate=(
            "Starting cash: $%{x:,.2f}<br>"
            "ROI: %{y:.1%}<extra></extra>"
        )
    )
    fig_roi.add_hline(
        y=0, line_dash="dash", line_color="#94a3b8", line_width=1,
        annotation_text="break-even (ROI = 0)", annotation_position="top right",
        annotation_font_color="#94a3b8", annotation_font_size=10,
    )

    # ── Per-trial supply slider ──
    slider_trials = sorted(terminal_df["trial_id"].unique())[:50]
    frames = []
    for tid in slider_trials:
        sub = (
            terminal_df[terminal_df.trial_id == tid]
            .pivot(index="i", columns="j", values="supply")
            .reindex(index=range(GRID_SIZE), columns=range(GRID_SIZE))
            .to_numpy()
        )
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=sub.tolist(),
                x=COL_LABELS, y=ROW_LABELS,
                colorscale="Viridis",
                hovertemplate=(
                    "<b>Score: Man City %{y} — Crystal Palace %{x}</b>"
                    "<br>Final OT supply: %{z:,.2f} units<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(text="OT supply<br>(units)", side="right", font=dict(size=11)),
                    thickness=12, tickfont=dict(size=10),
                ),
            )],
            name=str(tid),
        ))
    init_z = frames[0].data[0].z if frames else mean_supply.tolist()
    fig_slider = go.Figure(
        data=[go.Heatmap(
            z=init_z,
            x=COL_LABELS, y=ROW_LABELS,
            colorscale="Viridis",
            hovertemplate=(
                "<b>Score: Man City %{y} — Crystal Palace %{x}</b>"
                "<br>Final OT supply: %{z:,.2f} units<extra></extra>"
            ),
            colorbar=dict(
                title=dict(text="OT supply<br>(units)", side="right", font=dict(size=11)),
                thickness=12, tickfont=dict(size=10),
            ),
        )],
        frames=frames,
    )
    fig_slider.update_layout(
        title=dict(
            text=titled(
                "Final outcome-token supply, trial-by-trial",
                "Use the slider or ▶ Play to inspect one trial at a time (first 50 trials)",
            ),
            x=0.02, xanchor="left",
        ),
        height=470,
        margin=dict(l=70, r=20, t=100, b=90),
        xaxis=dict(
            title=dict(text="Crystal Palace — goals scored", font=dict(size=11)),
            tickmode="array", tickvals=COL_LABELS, ticktext=COL_LABELS,
            constrain="domain",
        ),
        yaxis=dict(
            title=dict(text="Manchester City — goals scored", font=dict(size=11)),
            tickmode="array", tickvals=ROW_LABELS, ticktext=ROW_LABELS,
            scaleanchor="x", scaleratio=1,
        ),
        sliders=[{
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate"}],
                 "label": f.name, "method": "animate"}
                for f in frames
            ],
            "currentvalue": {"prefix": "Showing trial #: ", "font": {"size": 12, "color": "#e2e8f0"}},
            "pad": {"t": 50},
        }] if frames else [],
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "y": 1.13,
            "x": 0.02,
            "xanchor": "left",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 300, "redraw": True},
                                 "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}]},
            ],
        }] if frames else [],
    )

    store["result"] = {
        "kpis": {
            "mean_pool": mean_pool,
            "mean_ppu": mean_ppu,
            "mean_pnl": mean_pnl,
            "median_roi": median_roi,
            "pct_pos": pct_pos,
            "mean_house_pnl": mean_house_pnl,
            "mean_gini": mean_gini,
            "n_trials": cfg.n_trials,
            "n_agents": cfg.n_agents,
            "conservation_error": cons["max_relative_error"],
        },
        "charts": {
            "supply": hm_json(
                mean_supply,
                title="Mean final outcome-token supply per cell",
                subtitle="Average units of OT minted into each scoreline by trial-end",
                colorbar_title="Mean OT supply<br>(units)",
                hover_value_label="Mean supply",
                fmt=",.2f",
                hover_suffix=" units",
            ),
            "winfreq": hm_json(
                win_freq,
                title="Empirical winner frequency per cell",
                subtitle="Share of trials in which each scoreline was drawn as the winner",
                colorbar_title="Winner<br>frequency",
                hover_value_label="Won as outcome in",
                colorscale="Blues",
                fmt=".1%",
                hover_suffix=" of trials",
            ),
            "payout": hm_json(
                mean_payout,
                title="Mean marginal payout multiplier per cell",
                subtitle="Expected payout per $1 minted at the current supply (4·x^¾ / 7·x ratio)",
                colorbar_title="Multiplier<br>(× per $1)",
                hover_value_label="Marginal payout",
                colorscale="Magma",
                fmt=".3f",
                hover_suffix="× per $1",
            ),
            "realized": hm_json(
                realized,
                title="Realised payout per OT, given the cell wins",
                subtitle="Average dollars paid per winning outcome token, conditional on this cell being drawn",
                colorbar_title="$ per winning<br>OT unit",
                hover_value_label="Payout if winner",
                colorscale="Reds",
                fmt=",.2f",
                hover_prefix="$",
                hover_suffix=" / unit",
            ),
            "pnl": fig_pnl.to_json(),
            "lorenz": fig_lorenz.to_json(),
            "roi": fig_roi.to_json(),
            "slider": fig_slider.to_json(),
        },
    }
    store["status"] = "done"
    store["progress"] = 100


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    data = request.json or {}

    def _int(k, default):
        try:
            return int(data.get(k, default))
        except (ValueError, TypeError):
            return default

    def _float(k, default):
        try:
            return float(data.get(k, default))
        except (ValueError, TypeError):
            return default

    winner_mode = data.get("winner_distribution", "realistic")
    strategy = data.get("agent_strategy", "uniform_random")

    try:
        cfg = SimConfig(
            n_agents=_int("n_agents", 100),
            balance_min=_float("balance_min", 50.0),
            balance_max=_float("balance_max", 5000.0),
            n_trials=_int("n_trials", 200),
            init_mcap_min=_float("init_mcap_min", 0.10),
            init_mcap_max=_float("init_mcap_max", 10.0),
            winner_mode=winner_mode,
            strategy=strategy,
            seed=_int("seed", 1234),
        )
    except ValueError as exc:
        return {"error": str(exc)}, 400

    run_id = str(uuid.uuid4())
    _runs[run_id] = {"status": "queued", "progress": 0, "result": None}
    t = threading.Thread(target=_run_simulation, args=(run_id, cfg), daemon=True)
    t.start()
    return {"run_id": run_id}


@app.route("/progress/<run_id>")
def progress(run_id: str):
    def _gen() -> Iterator[str]:
        while True:
            store = _runs.get(run_id)
            if store is None:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                return
            pct = store.get("progress", 0)
            msg = store.get("progress_message", f"{pct}%")
            status = store["status"]
            yield f"data: {json.dumps({'progress': pct, 'message': msg, 'status': status})}\n\n"
            if status == "done":
                return
            time.sleep(0.4)

    return Response(
        stream_with_context(_gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/result/<run_id>")
def result(run_id: str):
    store = _runs.get(run_id)
    if store is None:
        return {"error": "not found"}, 404
    if store["status"] != "done":
        return {"error": "not ready"}, 202
    return store["result"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
