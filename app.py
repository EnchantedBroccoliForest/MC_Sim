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

    def hm_json(z, title, colorscale="Viridis", fmt=".3f"):
        fig = go.Figure(
            data=go.Heatmap(
                z=z.tolist(),
                x=COL_LABELS,
                y=ROW_LABELS,
                colorscale=colorscale,
                hovertemplate="MCI %{y} − CRY %{x}<br>value: %{z:" + fmt + "}<extra></extra>",
            )
        ).update_layout(
            title=title,
            height=380,
            margin=dict(l=50, r=20, t=50, b=50),
            xaxis_title="CRY goals",
            yaxis_title="MCI goals",
        )
        return fig.to_json()

    # PnL histogram
    fig_pnl = px.histogram(
        agent_df, x="pnl", nbins=80, title="Agent P&L Distribution",
        color_discrete_sequence=["#4C78A8"],
    ).update_layout(height=350, margin=dict(l=50, r=20, t=50, b=50),
                    xaxis_title="P&L ($)", yaxis_title="Count")
    fig_pnl.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)

    # Lorenz curve
    pop, share = lorenz_curve(agent_df["terminal_cash"].to_numpy())
    fig_lorenz = go.Figure()
    fig_lorenz.add_trace(go.Scatter(x=pop.tolist(), y=share.tolist(), mode="lines",
                                    name="Terminal cash", line=dict(color="#4C78A8", width=2)))
    fig_lorenz.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                    name="Perfect equality", line=dict(color="black", dash="dash")))
    fig_lorenz.update_layout(title="Lorenz Curve — Agent Terminal Cash", height=350,
                              margin=dict(l=50, r=20, t=50, b=50),
                              xaxis_title="Cumulative share of agents",
                              yaxis_title="Cumulative share of cash")

    # ROI vs balance scatter (sampled)
    sample = agent_df.sample(min(5000, len(agent_df)), random_state=0)
    fig_roi = px.scatter(sample, x="starting_balance", y="roi", opacity=0.3,
                         title="ROI vs Starting Balance",
                         color_discrete_sequence=["#4C78A8"]).update_layout(
        height=350, margin=dict(l=50, r=20, t=50, b=50),
        xaxis_title="Starting balance ($)", yaxis_title="ROI")
    fig_roi.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)

    # Trial slider — first 50 trials
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
            data=[go.Heatmap(z=sub.tolist(), x=COL_LABELS, y=ROW_LABELS, colorscale="Viridis")],
            name=str(tid),
        ))
    init_z = frames[0].data[0].z if frames else mean_supply.tolist()
    fig_slider = go.Figure(
        data=[go.Heatmap(z=init_z, x=COL_LABELS, y=ROW_LABELS, colorscale="Viridis")],
        frames=frames,
    )
    fig_slider.update_layout(
        title="Per-Trial Final Supply (first 50 trials)",
        height=430,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis_title="CRY goals",
        yaxis_title="MCI goals",
        sliders=[{
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate"}],
                 "label": f.name, "method": "animate"}
                for f in frames
            ],
            "currentvalue": {"prefix": "Trial: "},
            "pad": {"t": 50},
        }] if frames else [],
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "y": 1.15,
            "x": 0.05,
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
            "supply": hm_json(mean_supply, "Mean Terminal OT Supply"),
            "payout": hm_json(mean_payout, "Mean Marginal Payout Multiplier", "Magma"),
            "realized": hm_json(realized, "Realized Payout / Unit (when cell wins)", "Reds", ".1f"),
            "winfreq": hm_json(win_freq, "Empirical Winner Frequency", "Blues", ".3f"),
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
