# MC_Sim — Dynamic Parimutuel Monte Carlo

Monte Carlo simulation of a 42.space-style dynamic parimutuel market over an
8×8 grid of scoreline outcomes (MCI vs. CRY, goals `0..6, ≥7`). Agents mint
outcome tokens against a bonding curve `p(x) = x^(3/4)` until cash runs out;
one cell is then drawn as the winner and the entire pool is paid pro-rata to
holders of the winning OT.

## Install

```
pip install -r requirements.txt
```

## Run

Default 1000-trial Monte Carlo with 100 agents, Poisson(1.6, 1.1) winner prior:

```
python -m parimutuel_sim.cli --seed 1234
```

Useful flags:

- `--n-trials S` / `--n-agents N`
- `--balance-min` / `--balance-max` (agent starting cash range)
- `--init-mcap-min` / `--init-mcap-max` (per-OT house seed range)
- `--winner-distribution uniform | realistic | fixed:i,j`
- `--agent-strategy uniform_random | weighted_by_marginal_payout`
- `--seed N`

Outputs land in `outputs/runs/<timestamp>/` with `REPORT.md`,
`summary.json`, parquet/CSV tables, static PNG plots, and an interactive
`dashboard.html`.

## Tests

```
python -m pytest parimutuel_sim/tests/ -v
```

## Layout

```
parimutuel_sim/
  market.py       # bonding-curve math + MarketState (with house seed)
  agents.py       # Agent + cell/amount selection
  settlement.py   # winner prior + pro-rata payout
  simulation.py   # SimConfig, run_one_trial, run_monte_carlo
  analytics.py    # Gini, Lorenz, percentiles, conservation, frame builders
  viz.py          # static plots + Plotly dashboard
  cli.py          # argparse entry; writes all artifacts + REPORT.md
  tests/test_market.py
```
