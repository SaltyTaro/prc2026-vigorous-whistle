"""Heathrow group-level reconciliation with the CAA punctuality statistics.

The CAA publishes, per month, reporting airport, airline and origin/destination, the number of matched flights and the average
departure delay in whole minutes, delay = actual off-block minus the planned gate time, early flights counted as zero. For our rows
of the same group the delay is a - y (schedule interval minus taxi time), so the group's published mean constrains the group's mean
taxi prediction, exactly as the daily totals constrain an airport-day (src/prc2026_daily), at a coarser precision (the minute).

Procedure for one month (label-free on the read month):
  1. groups = (CAA airline, CAA destination) of the rows that map (docs/caa_heathrow/airlines.csv, destinations.csv), cargo excluded;
     a group is read when the published count is at least `min_group` and our count agrees with it within max(2, 3%).
  2. implied mean of max(delay, 0) under the row error: P(s) = mean_i [ m_i Phi(m_i / s_i) + s_i phi(m_i / s_i) ], m_i = (a_i - y_i - s)/60,
     s_i = the row error sd of the row's class from the other months; the published mean carries a definitional offset delta
     (published minus the label mean, from the other months' labels), so the level error is e = P(0) + delta - D, clipped to +-clip_min.
  3. noise of e: rounding (1/12 min^2), row noise (s^2 / n), definitional noise (from the other months); the group biases' variance B^2
     is the month's own dispersion of e beyond its noise (empirical Bayes, label-free); the shrunk correction is e B^2 / (B^2 + noise).
  4. the group's taxi shift s solves P(s) = P(0) - correction; rows of the group below `max_pred_s` move by s; floor -12 s.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt

import numpy as np
import pandas as pd

FLOOR = -12.0
ROUNDING_VAR = 1.0 / 12.0   # variance of a uniform rounding error of +-0.5 minute


@dataclass(frozen=True)
class Settings:
    min_group: int = 30
    count_tolerance: float = 0.03
    clip_min: float = 2.0
    max_pred_s: float = 10800.0
    min_prior_var: float = 0.01   # min^2


def _phi(x):
    return np.exp(-0.5 * x * x) / sqrt(2 * pi)


def _Phi(x):
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2)))


def implied_mean(m_min: np.ndarray, sd_min: np.ndarray, shift_min: float = 0.0) -> float:
    """Mean over the rows of E[max(delay, 0)] with delay ~ N(m - shift, sd^2), in minutes."""
    m = m_min - shift_min
    z = m / sd_min
    return float(np.mean(m * _Phi(z) + sd_min * _phi(z)))


def solve_shift(m_min: np.ndarray, sd_min: np.ndarray, target: float, lo: float = -30.0, hi: float = 30.0) -> float:
    """The shift (minutes of taxi) whose implied mean equals the target; bisection, P is decreasing in the shift."""
    f_lo, f_hi = implied_mean(m_min, sd_min, lo) - target, implied_mean(m_min, sd_min, hi) - target
    if f_lo < 0:
        return lo
    if f_hi > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if implied_mean(m_min, sd_min, mid) - target > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reconcile_month(rows: pd.DataFrame, published: pd.DataFrame, sd_by_class: dict, delta_min: float, def_var: float,
                    settings: Settings = Settings()) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """rows: one month of Heathrow departures with columns a (s), pred (s), cls (key of sd_by_class), airline, dest (mapped names or NaN).
    published: index (airline, dest) with columns n_pub, delay_min. Returns the new predictions, the group table and a summary."""
    new = rows["pred"].to_numpy(dtype=float).copy()
    ok = rows["airline"].notna() & rows["dest"].notna()
    groups = []
    for (airline, dest), g in rows[ok].groupby(["airline", "dest"], sort=True):
        if (airline, dest) not in published.index:
            continue
        n_pub, d_pub = published.loc[(airline, dest), ["n_pub", "delay_min"]]
        n_pub = int(n_pub)
        if n_pub < settings.min_group or abs(len(g) - n_pub) > max(2, settings.count_tolerance * n_pub):
            continue
        use = g[g["pred"] < settings.max_pred_s]
        if len(use) < settings.min_group // 2:
            continue
        m = (use["a"].to_numpy(dtype=float) - use["pred"].to_numpy(dtype=float)) / 60.0
        sd = np.array([sd_by_class[c] for c in use["cls"]], dtype=float) / 60.0
        p0 = implied_mean(m, sd)
        e = float(np.clip(p0 + delta_min - float(d_pub), -settings.clip_min, settings.clip_min))
        noise = ROUNDING_VAR + float(np.mean(sd ** 2)) / len(use) + def_var
        groups.append({"airline": airline, "dest": dest, "n_pub": n_pub, "n_ours": int(len(g)), "n_used": int(len(use)), "published_min": float(d_pub), "implied_min": p0,
                       "level_error_min": e, "noise_var": noise, "positions": use.index.to_numpy(), "m": m, "sd": sd})
    if not groups:
        return new, pd.DataFrame(), {"groups_read": 0, "prior_var_min2": 0.0}
    n = np.array([g["n_used"] for g in groups], dtype=float)
    e = np.array([g["level_error_min"] for g in groups])
    noise = np.array([g["noise_var"] for g in groups])
    prior_var = max(float(np.sum(n * (e ** 2 - noise)) / n.sum()), settings.min_prior_var)
    for g in groups:
        g["correction_min"] = g["level_error_min"] * prior_var / (prior_var + g["noise_var"])
        s = solve_shift(g["m"], g["sd"], g["implied_min"] - g["correction_min"])
        g["shift_s"] = 60.0 * s
        new[g["positions"]] = np.maximum(new[g["positions"]] + g["shift_s"], FLOOR)
    table = pd.DataFrame([{k: v for k, v in g.items() if k not in ("positions", "m", "sd")} for g in groups])
    summary = {"groups_read": len(groups), "rows_moved": int(sum(g["n_used"] for g in groups)), "prior_var_min2": prior_var, "prior_sd_s": 60.0 * sqrt(prior_var),
               "mean_abs_shift_s": float(table["shift_s"].abs().mean()), "max_abs_shift_s": float(table["shift_s"].abs().max()),
               "mean_noise_sd_s": float(60.0 * np.sqrt(table["noise_var"]).mean())}
    return new, table, summary
