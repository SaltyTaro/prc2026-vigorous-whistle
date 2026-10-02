"""The published daily figure's own definition, as the noise model of the daily-totals reconciliation (src/prc2026_daily).

EUROCONTROL's All_Pre-Departure_Delay (DLY_ALL_PRE_2, minutes; FLT_DEP_IFR_2, departures) against the supplied 2025 clocks, per airport
and UTC off-block day (tools/scout_published_definition.py, docs/scout/published_definition.json):

  - DLY_ALL_PRE_2 x 60 = sum over departures of max(round((block - schedule) / 60), 0) x 60, exactly, on every day whose departure count
    equals ours, at EGLL, LIRF, EDDM, LEMD, LTFM, LFPG and LEBL; within about 70-90 s a day (exact seconds plus a constant) at EDDF,
    EHAM and LSZH.
  - EHAM, LSZH and LEBL count IFR departures only (FLIGHT_RULE_mvt == "I"). With VFR rows in the arithmetic, as in v48, EHAM's figure
    looks 2,300 s a day noisy and 86% of its days off by more than ten minutes, so v48 nearly ignored it.
  - What is left comes with the count difference dn = published count - our rows: a published flight missing from the supplied rows
    (dn > 0, mostly LFPG, about 1,300-2,000 s each) adds its delay, a row of ours missing from the figure (dn < 0) takes its delay away.

Model of the published total S of an airport-day, given the day's own total T (the sum the labels would give over the rows that enter
the figure) and dn, both label-free in the scored months except T:

    S = T + c0 + mu_s |dn| + e,   e ~ (1 - w) N(0, s0^2 + |dn| s_s^2) + w/2 N(0, 3600^2) + w/2 N(0, 28800^2),   s = sign(dn)

c0 and s0 from the fitting months' days with dn = 0, mu_s and s_s from their days with that sign of dn (pooled over airports where an
airport has too few such days), w = the share of fitting days outside four core sd (at least 0.02). Nothing is fitted on a score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IFR_ONLY = ("EHAM", "LSZH", "LEBL")
MIN_ZERO_DAYS, MIN_SIGN_DAYS = 5, 10
FLOOR_S, WIDE_K, WIDE_FLOOR = 30.0, 4.0, 0.02
POOLED_DEFAULT = {+1: (0.0, 1000.0), -1: (0.0, 1000.0)}


def robust_sd(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if len(v) else 0.0


def in_figure(airport: np.ndarray, ifr: np.ndarray) -> np.ndarray:
    """True for the rows that enter the published figure: every row, except non-IFR rows at the IFR-only airports."""
    return ~(np.isin(np.asarray(airport), IFR_ONLY) & ~np.asarray(ifr, dtype=bool))


def fit(days: pd.DataFrame) -> dict:
    """Per airport, from fitting days with columns airport, diff (published minus own total, s) and dn (published count minus rows)."""
    model, pooled = {}, {+1: [], -1: []}
    for airport, g in days.groupby("airport", sort=True):
        zero = g[g["dn"] == 0]
        base = zero if len(zero) >= MIN_ZERO_DAYS else g
        c0, s0 = float(np.median(base["diff"])), max(robust_sd(base["diff"].to_numpy()), FLOOR_S)
        m = {"c0": c0, "s0": s0, "days": int(len(g)), "zero_days": int(len(zero)), "sign": {}}
        for sign in (+1, -1):
            x = g[np.sign(g["dn"]) == sign]
            if len(x) >= MIN_SIGN_DAYS:
                k = (sign * x["dn"]).to_numpy(dtype=float)
                mu = float((x["diff"].to_numpy() - c0).sum() / k.sum())
                sd = max(robust_sd(((x["diff"].to_numpy() - c0) - mu * k) / np.sqrt(k)), FLOOR_S)
                m["sign"][sign] = {"mu": mu, "sd": sd, "days": int(len(x)), "pooled": False}
                pooled[sign].append((mu, sd))
        model[airport] = m
    for sign in (+1, -1):
        mu, sd = (float(np.median([v[0] for v in pooled[sign]])), float(np.median([v[1] for v in pooled[sign]]))) if pooled[sign] else POOLED_DEFAULT[sign]
        for m in model.values():
            m["sign"].setdefault(sign, {"mu": mu, "sd": sd, "days": 0, "pooled": True})
    for airport, g in days.groupby("airport", sort=True):
        m = model[airport]
        offset, sd = zip(*(day_terms_core(m, float(dn)) for dn in g["dn"].to_numpy()))
        resid = g["diff"].to_numpy(dtype=float) - np.asarray(offset)
        m["wide"] = max(float((np.abs(resid) > WIDE_K * np.asarray(sd)).mean()), WIDE_FLOOR)
    return model


def day_terms_core(m: dict, dn: float) -> tuple[float, float]:
    if dn == 0:
        return m["c0"], m["s0"]
    s = m["sign"][+1 if dn > 0 else -1]
    return m["c0"] + s["mu"] * abs(dn), float(np.sqrt(m["s0"] ** 2 + abs(dn) * s["sd"] ** 2))


def day_terms(model: dict, airport: str, dn: float) -> tuple[float, float, float]:
    """(expected published minus own total, core noise sd, wide share) of one airport-day."""
    m = model[airport]
    offset, sd = day_terms_core(m, float(dn))
    return offset, sd, m["wide"]


def fitting_days(fit_rows: pd.DataFrame, share: float = 0.9) -> pd.DataFrame:
    """Fitting airport-days from rows with airport, block_day, bday, pos_true, a, pub, n_pub: the own total over the rows (true off-block
    day), the published total and count; days whose rows cover at least `share` of the published count."""
    own = fit_rows.groupby(["airport", "block_day"]).agg(true=("pos_true", "sum"), rows=("a", "size")).rename_axis(["airport", "bday"])
    pub = fit_rows.groupby(["airport", "bday"]).agg(pub=("pub", "first"), n_pub=("n_pub", "first"))
    own = own.join(pub, how="inner").dropna()
    own = own[own["rows"] >= share * own["n_pub"]]
    return pd.DataFrame({"airport": own.index.get_level_values("airport"), "bday": own.index.get_level_values("bday"),
                         "diff": (own["pub"] - own["true"]).to_numpy(), "dn": (own["n_pub"] - own["rows"]).to_numpy()})
