"""Neighbour-state columns for the guarded stacker (configs/stack_nbr_v1.json).

Input table t: airport, ts (take-off, seconds), date (UTC take-off date), matched (bool), value (the pre-stacker chain value 'new'),
v9. Neighbours are matched rows of the same airport, the row itself excluded; ex = value - v9.

  nx  = run_nbr_state.state (mean of ex over neighbours within 900 s, at least 3, minus the airport-day mean of ex over matched rows)
  nxk = kernel form: (sum w ex + 3 * daymean) / (sum w + 3) - daymean over neighbours within 1,800 s, w = exp(-|dt| / 900 s), at
        least 3 neighbours (the coordinator's scratch ref2.state3 with tau 900, shrink 3)
  nn  = number of neighbours within 900 s
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_nbr_state as N  # noqa: E402

COLS = ["nx", "nxk", "nn"]
TAU, SHRINK, WK, MIN_N = 900.0, 3.0, 1800.0, 3


def columns(t: pd.DataFrame) -> pd.DataFrame:
    t = t.reset_index(drop=True)
    ex = (t["value"] - t["v9"]).to_numpy(dtype=float)
    nx = N.state(t)
    nxk, nn = np.full(len(t), np.nan), np.zeros(len(t))
    for _, g in t.groupby("airport", sort=False):
        gi = g.index.to_numpy()
        mt = g["matched"].to_numpy(dtype=bool) & np.isfinite(ex[gi])
        gm = g[mt].sort_values("ts")
        tg, eg = gm["ts"].to_numpy(dtype=float), ex[gm.index.to_numpy()]
        base = g["date"].map(pd.Series(ex[gi][mt]).groupby(g["date"].to_numpy()[mt]).mean()).to_numpy(dtype=float)
        ts = g["ts"].to_numpy(dtype=float)
        lo9, hi9 = np.searchsorted(tg, ts - N.WINDOW, side="left"), np.searchsorted(tg, ts + N.WINDOW, side="right")
        nn[gi] = (hi9 - lo9) - mt
        lo, hi = np.searchsorted(tg, ts - WK, side="left"), np.searchsorted(tg, ts + WK, side="right")
        sw, se, k = np.zeros(len(ts)), np.zeros(len(ts)), np.zeros(len(ts))
        own = ex[gi]
        for off in range(int((hi - lo).max()) if len(ts) else 0):
            idx = lo + off
            ok = idx < hi
            j = np.where(ok, idx, 0)
            w = np.where(ok, np.exp(-np.abs(tg[j] - ts) / TAU), 0.0)
            selfm = ok & mt & (tg[j] == ts) & (eg[j] == own)
            w = np.where(selfm, 0.0, w)
            sw += w
            se += w * np.where(ok, eg[j], 0.0)
            k += ok & ~selfm
        nxk[gi] = np.where(k >= MIN_N, (se + SHRINK * base) / (sw + SHRINK), np.nan) - base
    return pd.DataFrame({"nx": nx, "nxk": nxk, "nn": nn}).astype(np.float32)
