"""Extreme-day calibration of the reconciliation's day shift (a layer after v48's procedure; configs/scout_extreme_shift_v1.json).

v48's posterior day shift has a Gaussian prior N(0, tau^2) with tau fitted on ordinary days, so on a day whose published total implies a
large shift it applies little of it (docs/scout/README.md). This layer replaces v48's day shift by beta_a * I_d on such days only:

  I_d       = -(published - c0_a - implied total) / n_eff over the rows that enter the published figure (IFR only at EHAM / LSZH / LEBL),
              implied total = sum of max(a - prediction, 0), n_eff = the number of those rows with a > prediction
  c0_a      = median over fitting days of (published - the labels' own total)
  beta_a    = least-squares slope (through zero) of the day's mean residual of in-figure, v48-moved, ordinary (|r| < 1800 s) rows on I_d,
              clipped to [0, 1]; corr_a the same pair's correlation; both over fitting days whose counts agree (dn = 0), n_eff >= 100
  airports  with corr_a >= CORR_MIN over at least MIN_DAYS fitting days
  days      |I_d| >= I_MIN_S, n_eff >= 100, and dn = 0 or "sign-safe": a published flight we lack (dn > 0) can only raise the published
            total, a row of ours the figure lacks (dn < 0) only the implied total, so the day qualifies when that cannot have produced the
            discrepancy's sign (dn > 0 with published below implied, dn < 0 with published above implied)
  rows      in the figure and moved by v48 (heavy rows and rows of unread days keep v48): v48 value - v48 day shift + beta_a * I_d,
            floor -12 s
No label of a day read enters; labels enter only the fitting days' need and own totals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

I_MIN_S, CORR_MIN, MIN_DAYS, MIN_NEFF, NEED_CLIP_S, FLOOR_S = 80.0, 0.9, 30, 100, 1800.0, -12.0


def day_frame(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per day (keys, e.g. ["month", "airport", "bday"]): rows, implied, n_eff, pub, n_pub, dn, delta_raw, eligible, fit_ok; with labels
    also true_own (own total by true off-block day) and need. Rows need: airport, bday, a, stack, pub, n_pub, in_figure, moved, extra;
    with labels: label, block_day, pos_true."""
    f = rows[rows["in_figure"]].assign(pos_hat=lambda x: np.maximum(x["a"] - x["stack"], 0.0), pos_d=lambda x: (x["a"] > x["stack"]).astype(float))
    day = f.groupby(keys).agg(rows=("a", "size"), implied=("pos_hat", "sum"), n_eff=("pos_d", "sum"), pub=("pub", "first"), n_pub=("n_pub", "first"))
    if "label" in rows:
        fl = f[~f["extra"]]
        own_keys = [k if k != "bday" else "block_day" for k in keys]
        day = day.join(fl.groupby(own_keys)["pos_true"].sum().rename_axis(keys).rename("true_own"))
        o = fl[fl["moved"] & ((fl["label"] - fl["stack"]).abs() < NEED_CLIP_S)]
        day = day.join((o["label"] - o["stack"]).groupby([o[k] for k in keys]).mean().rename_axis(keys).rename("need"))
    day["dn"] = day["n_pub"] - day["rows"]
    day["delta_raw"] = day["pub"] - day["implied"]
    base = (day["n_eff"] >= MIN_NEFF) & day["pub"].notna()
    day["fit_ok"] = base & (day["dn"] == 0)
    day["eligible"] = base & ((day["dn"] == 0) | ((day["dn"] > 0) & (day["delta_raw"] < 0)) | ((day["dn"] < 0) & (day["delta_raw"] > 0)))
    return day


def implied_shift(day: pd.DataFrame, c0: pd.Series) -> pd.Series:
    airport = day.index.get_level_values("airport")
    return -(day["pub"] - airport.map(c0).to_numpy(dtype=float) - day["implied"]) / day["n_eff"]


def fit(days: pd.DataFrame) -> dict:
    """c0, per-airport statistics and the calibrated airports' beta from fitting days (fit_ok, with need and true_own)."""
    d = days[days["fit_ok"]].dropna(subset=["need", "true_own"])
    c0 = (d["pub"] - d["true_own"]).groupby(level="airport").median()
    d = d.assign(I=implied_shift(d, c0))
    stats = {}
    for airport, x in d.groupby(level="airport"):
        stats[airport] = {"days": int(len(x)), "corr": float(np.corrcoef(x["need"], x["I"])[0, 1]) if len(x) > 2 else float("nan"),
                          "beta_raw": float((x["need"] * x["I"]).sum() / (x["I"] ** 2).sum())}
    beta = {a: float(np.clip(s["beta_raw"], 0.0, 1.0)) for a, s in stats.items() if s["days"] >= MIN_DAYS and s["corr"] >= CORR_MIN}
    return {"c0": {a: float(v) for a, v in c0.items()}, "stats": stats, "beta": beta}


def shifts(days: pd.DataFrame, params: dict) -> pd.DataFrame:
    """The days the layer changes, with I_d and the new shift."""
    d = days[days["eligible"]]
    d = d[d.index.get_level_values("airport").isin(list(params["beta"]))]
    d = d.assign(I=implied_shift(d, pd.Series(params["c0"])))
    d = d[d["I"].abs() >= I_MIN_S]
    return d.assign(shift_new=d.index.get_level_values("airport").map(params["beta"]).to_numpy(dtype=float) * d["I"])[["I", "shift_new", "dn", "rows", "n_eff"]]


def apply(rows: pd.DataFrame, changed_days: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(new values, mask of changed rows). Rows need v48 (reconciled value), v48_shift (its day shift), moved, in_figure."""
    t = rows.join(changed_days[["shift_new"]], on=keys)
    mask = (t["moved"] & t["in_figure"] & t["shift_new"].notna()).to_numpy()
    new = t["v48"].to_numpy(dtype=float).copy()
    new[mask] = np.maximum(t["v48"].to_numpy(dtype=float)[mask] - t["v48_shift"].fillna(0.0).to_numpy(dtype=float)[mask] + t["shift_new"].to_numpy(dtype=float)[mask], FLOOR_S)
    return new, mask
