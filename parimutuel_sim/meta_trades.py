"""Meta-trade abstraction over the 8x8 OT grid.

A meta trade buys into a named *bucket* of underlying OT cells in a single
agent action. The bucket weights are snapshotted from per-cell market caps
*before* any leg is minted, so the result is independent of leg-iteration
order and not vulnerable to within-trade rebalancing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

log = logging.getLogger(__name__)

# (mci_idx, cry_idx), both in 0..GRID_SIZE-1
GridIndex = tuple[int, int]


@dataclass(frozen=True)
class MetaTradeDef:
    key: str                       # e.g. "MCI_WIN"
    set_name: str                  # e.g. "moneyline"
    display_name: str              # e.g. "MCI Win"
    cells: tuple[GridIndex, ...]   # underlying OT cells


def _moneyline_buckets(n: int = 8) -> list[MetaTradeDef]:
    mci_cells = tuple((i, j) for i in range(n) for j in range(n) if i > j)
    draw_cells = tuple((i, i) for i in range(n))
    cry_cells = tuple((i, j) for i in range(n) for j in range(n) if j > i)
    return [
        MetaTradeDef("MCI_WIN", "moneyline", "MCI Win", mci_cells),
        MetaTradeDef("DRAW",    "moneyline", "Draw",    draw_cells),
        MetaTradeDef("CRY_WIN", "moneyline", "CRY Win", cry_cells),
    ]


def _spread_buckets(n: int = 8) -> list[MetaTradeDef]:
    """Typical soccer spreads (e.g., +/- 1.5 goals)."""
    return [
        MetaTradeDef("MCI_MINUS_1_5", "spread", "MCI -1.5", tuple((i, j) for i in range(n) for j in range(n) if i >= j + 2)),
        MetaTradeDef("CRY_PLUS_1_5", "spread", "CRY +1.5", tuple((i, j) for i in range(n) for j in range(n) if j >= i - 1)),
        MetaTradeDef("CRY_MINUS_1_5", "spread", "CRY -1.5", tuple((i, j) for i in range(n) for j in range(n) if j >= i + 2)),
        MetaTradeDef("MCI_PLUS_1_5", "spread", "MCI +1.5", tuple((i, j) for i in range(n) for j in range(n) if i >= j - 1)),
    ]


def _totals_buckets(n: int = 8) -> list[MetaTradeDef]:
    """Typical soccer totals (e.g., Over/Under 2.5 goals)."""
    return [
        MetaTradeDef("OVER_1_5", "totals", "Over 1.5", tuple((i, j) for i in range(n) for j in range(n) if i + j >= 2)),
        MetaTradeDef("UNDER_1_5", "totals", "Under 1.5", tuple((i, j) for i in range(n) for j in range(n) if i + j <= 1)),
        MetaTradeDef("OVER_2_5", "totals", "Over 2.5", tuple((i, j) for i in range(n) for j in range(n) if i + j >= 3)),
        MetaTradeDef("UNDER_2_5", "totals", "Under 2.5", tuple((i, j) for i in range(n) for j in range(n) if i + j <= 2)),
        MetaTradeDef("OVER_3_5", "totals", "Over 3.5", tuple((i, j) for i in range(n) for j in range(n) if i + j >= 4)),
        MetaTradeDef("UNDER_3_5", "totals", "Under 3.5", tuple((i, j) for i in range(n) for j in range(n) if i + j <= 3)),
    ]


def _exact_score_buckets(n: int = 8) -> list[MetaTradeDef]:
    """Every exact score combination as a meta trade."""
    buckets = []
    for i in range(n):
        for j in range(n):
            disp_i = "≥7" if i == n - 1 else str(i)
            disp_j = "≥7" if j == n - 1 else str(j)
            key = f"EXACT_{i}_{j}"
            name = f"Exact Score {disp_i}-{disp_j}"
            buckets.append(MetaTradeDef(key, "exact_score", name, ((i, j),)))
    return buckets


# Registry exposed to the rest of the sim. Keep this data-driven so future
# sets (Total Goals, Winning Margin, Exact Score, ...) can be added without
# touching call sites.
_all_meta_trades = (
    _moneyline_buckets()
    + _spread_buckets()
    + _totals_buckets()
    + _exact_score_buckets()
)

META_TRADES: dict[str, MetaTradeDef] = {m.key: m for m in _all_meta_trades}

MONEYLINE_KEYS: tuple[str, ...] = tuple(m.key for m in _moneyline_buckets())
SPREAD_KEYS: tuple[str, ...] = tuple(m.key for m in _spread_buckets())
TOTALS_KEYS: tuple[str, ...] = tuple(m.key for m in _totals_buckets())
EXACT_SCORE_KEYS: tuple[str, ...] = tuple(m.key for m in _exact_score_buckets())
ALL_META_KEYS: tuple[str, ...] = tuple(META_TRADES.keys())


def allocate(
    cash: float,
    cells: Sequence[GridIndex],
    market_caps: Mapping[GridIndex, float],
) -> dict[GridIndex, float]:
    """Split ``cash`` across ``cells`` weighted by a pre-trade market-cap snapshot.

    Returns a mapping ``cell -> dollars to mint into that cell``. The last leg
    absorbs any float dust so ``sum(values) == cash`` exactly. Empty-bucket
    fallback is equal weights (with a warning).
    """
    if not cells:
        return {}
    if cash <= 0:
        return {c: 0.0 for c in cells}

    caps = [max(0.0, float(market_caps[c])) for c in cells]
    total = sum(caps)
    if total <= 0:
        log.warning(
            "meta_trade: bucket total market cap is zero, falling back to equal weights"
        )
        share = cash / len(cells)
        alloc = {c: share for c in cells}
    else:
        alloc = {c: cash * (cap / total) for c, cap in zip(cells, caps)}

    keys = list(alloc.keys())
    running = sum(alloc[k] for k in keys[:-1])
    alloc[keys[-1]] = cash - running
    return alloc
