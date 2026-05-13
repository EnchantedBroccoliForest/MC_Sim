"""Bonding-curve market primitives for the parimutuel simulator.

Curve: p(x) = x^(3/4).
mcap(x) = (4/7) * x^(7/4).
Inverse-mint: x2 = ( x1^(7/4) + (7/4) * D )^(4/7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

import numpy as np

from . import meta_trades as _meta_trades_mod

GRID_SIZE = 8
N_CELLS = GRID_SIZE * GRID_SIZE
ALPHA = 3.0 / 4.0  # price curve exponent
EXP_OUT = 7.0 / 4.0  # 1 + alpha
INV_EXP = 4.0 / 7.0  # 1 / (1 + alpha)

# Separate RNG stream constant for seeding the initial grid; XORed with the
# user-supplied seed so changing agent count doesn't reshuffle the grid.
SEED_RNG_SALT = 0xA5A5
MCAP_TOTAL_RESYNC_INTERVAL = 2048


def marginal_price(x):
    """p(x) = x^(3/4). Vectorized."""
    x = np.asarray(x, dtype=float)
    return np.power(x, ALPHA, where=x > 0, out=np.zeros_like(x))


def marginal_price_scalar(x: float) -> float:
    """Scalar p(x) helper for inner loops."""
    return x**ALPHA if x > 0 else 0.0


def mcap(x):
    """Market cap = total USD ever spent minting this OT."""
    x = np.asarray(x, dtype=float)
    return (4.0 / 7.0) * np.power(x, EXP_OUT, where=x > 0, out=np.zeros_like(x))


def mcap_scalar(x: float) -> float:
    """Scalar market-cap helper for inner loops."""
    return (4.0 / 7.0) * x**EXP_OUT if x > 0 else 0.0


def marginal_payout(x):
    """Per-OT display multiplier = mcap(x) / p(x) = (4/7) * x."""
    return (4.0 / 7.0) * np.asarray(x, dtype=float)


def supply_for_mcap(m: float) -> float:
    """Inverse of mcap: solve (4/7) * x^(7/4) = m for x."""
    if m <= 0:
        return 0.0
    return ((7.0 / 4.0) * m) ** INV_EXP


def cost_to_mint(x1: float, x2: float) -> float:
    """Integral of p from x1 to x2."""
    return (4.0 / 7.0) * (x2**EXP_OUT - x1**EXP_OUT)


def mint_units(x1: float, dollars: float) -> Tuple[float, float]:
    """Given current supply x1 and $D to spend, return (x2, units_minted)."""
    if dollars <= 0:
        return x1, 0.0
    x2 = (x1**EXP_OUT + (7.0 / 4.0) * dollars) ** INV_EXP
    return x2, x2 - x1


@dataclass
class MetaTradeFill:
    """Record returned by ``MarketState.mint_meta`` describing one meta trade.

    ``legs`` carries one tuple per underlying cell touched, in the order they
    were minted. Each leg tuple is ``(cell, cash, units, pre_mcap, post_mcap)``.
    """

    meta_key: str
    agent_id: Optional[int]
    total_cash: float
    legs: List[Tuple[Tuple[int, int], float, float, float, float]]
    trial_id: Optional[int] = None
    tick: Optional[int] = None

    @property
    def total_units(self) -> float:
        return sum(units for _, _, units, _, _ in self.legs)


@dataclass
class MarketState:
    """Holds the live 8x8 supply grid and the initial seed grid for accounting."""

    init_mcap_min: float = 0.10
    init_mcap_max: float = 10.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    supply: np.ndarray = field(init=False)
    init_mcap: np.ndarray = field(init=False)
    init_supply: np.ndarray = field(init=False)
    _mcap_grid: np.ndarray = field(init=False, repr=False)
    _total_mcap: float = field(init=False, repr=False)
    _mints_since_total_resync: int = field(init=False, repr=False)
    meta_trade_log: List[MetaTradeFill] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.init_mcap_min < 0 or self.init_mcap_max < 0:
            raise ValueError("init_mcap_min and init_mcap_max must be >= 0")
        if self.init_mcap_max < self.init_mcap_min:
            raise ValueError("init_mcap_max must be >= init_mcap_min")
        if self.init_mcap_max == 0.0 and self.init_mcap_min == 0.0:
            # Empty market mode (spec §6.2 fallback).
            self.init_mcap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
            self.init_supply = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
        else:
            self.init_mcap = self.rng.uniform(
                self.init_mcap_min, self.init_mcap_max, size=(GRID_SIZE, GRID_SIZE)
            )
            self.init_supply = ((7.0 / 4.0) * self.init_mcap) ** INV_EXP
        self.supply = self.init_supply.copy()
        self.refresh_mcap_cache()

    def refresh_mcap_cache(self) -> None:
        """Rebuild cached market caps from the public supply grid."""
        self._mcap_grid = mcap(self.supply)
        self._total_mcap = float(self._mcap_grid.sum())
        self._mints_since_total_resync = 0

    def _resync_total_mcap(self) -> None:
        self._total_mcap = float(self._mcap_grid.sum())
        self._mints_since_total_resync = 0

    @property
    def mcap_grid(self) -> np.ndarray:
        # ``supply`` is public and existing callers may mutate it directly.
        self.refresh_mcap_cache()
        return self._mcap_grid.copy()

    @property
    def total_mcap(self) -> float:
        return self._total_mcap

    @property
    def price_grid(self) -> np.ndarray:
        return marginal_price(self.supply)

    @property
    def payout_grid(self) -> np.ndarray:
        return marginal_payout(self.supply)

    def mint(self, cell: Tuple[int, int], dollars: float) -> float:
        """Mint into one cell. Returns units minted."""
        i, j = cell
        x_old = float(self.supply[i, j])
        x_new, units = mint_units(x_old, dollars)
        if x_new != x_old:
            old_mcap = float(self._mcap_grid[i, j])
            new_mcap = mcap_scalar(float(x_new))
            self.supply[i, j] = x_new
            self._mcap_grid[i, j] = new_mcap
            self._total_mcap += new_mcap - old_mcap
            self._mints_since_total_resync += 1
            if self._mints_since_total_resync >= MCAP_TOTAL_RESYNC_INTERVAL:
                self._resync_total_mcap()
        return units

    def mcap_snapshot(self, cells: Tuple[Tuple[int, int], ...]) -> dict[Tuple[int, int], float]:
        """Return cached market caps for a set of cells."""
        return {c: float(self._mcap_grid[c]) for c in cells}

    def mint_meta(
        self,
        meta_key: str,
        dollars: float,
        agent_id: Optional[int] = None,
        pre_caps: Optional[Mapping[Tuple[int, int], float]] = None,
        allocation: Optional[Mapping[Tuple[int, int], float]] = None,
    ) -> MetaTradeFill:
        """Buy a named bucket of cells with ``dollars`` of cash.

        Snapshot the pre-trade market cap of each cell in the bucket, split
        ``dollars`` across them weighted by those caps, and mint each leg via
        the single-cell ``mint`` method (same bonding curve, no joint math).
        Records the fill in ``self.meta_trade_log``.
        """
        try:
            mdef = _meta_trades_mod.META_TRADES[meta_key]
        except KeyError as exc:
            raise ValueError(f"Unknown meta_key: {meta_key!r}") from exc

        cells = mdef.cells
        if pre_caps is None:
            pre_caps = self.mcap_snapshot(cells)
        if allocation is None:
            allocation = _meta_trades_mod.allocate(dollars, cells, pre_caps)

        legs: List[Tuple[Tuple[int, int], float, float, float, float]] = []
        for c in cells:
            cash_leg = float(allocation.get(c, 0.0))
            pre = float(pre_caps[c])
            units = float(self.mint(c, cash_leg)) if cash_leg > 0 else 0.0
            post = float(self._mcap_grid[c])
            legs.append((c, cash_leg, units, pre, post))

        fill = MetaTradeFill(
            meta_key=meta_key,
            agent_id=agent_id,
            total_cash=float(dollars) if dollars > 0 else 0.0,
            legs=legs,
        )
        self.meta_trade_log.append(fill)
        return fill

    @property
    def house_seed_total(self) -> float:
        return float(self.init_mcap.sum())
