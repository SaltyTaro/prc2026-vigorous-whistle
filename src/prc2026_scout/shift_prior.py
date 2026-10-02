"""A heavier-tailed prior for the reconciliation's day shift (src/prc2026_daily/reconcile.solve_day uses N(0, tau^2)).

Why. On the 2025 development months the labels confirm about 0.6 of the day shift that the published total implies (implied = minus the
discrepancy over n_eff; least-squares slope of the moved ordinary rows' mean residual on it), while v48's posterior applies about 0.4, and
only 0.13 against 0.41 on the days whose implied shift exceeds 80 s (scratch exploration, docs/scout/README.md). The Gaussian prior with
tau fitted on ordinary days caps the shift at a few tau, so a disruption day keeps most of its discrepancy.

Prior. b ~ (1 - pi) N(0, tau^2) + pi N(0, (k tau)^2), tau per airport as in v48, pi and k pooled over airports by maximum likelihood on
the fitting months' airport-day mean residuals (rows within `clip_s` of the prediction, days of at least 100 such rows), each standardised
by its airport's tau, with its sampling variance carried: z_d ~ (1 - pi) N(0, 1 + v_d) + pi N(0, k^2 + v_d).

solve_day_mix is solve_day's one-component path with that prior: the shift posterior on a grid spanning the wide component, and the
first-order tilt from the total's density under the mixture. Heavy rows keep their prediction (v48's settings: heavy_updates False).
With pi = 0, k = 1 and the grid of solve_day it reproduces solve_day exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026_daily.reconcile import GRID_S, N, Prepared, Settings, _at, _gauss_cf

PI_GRID = np.round(np.arange(0.01, 0.41, 0.01), 2)
K_GRID = np.round(np.arange(1.25, 10.01, 0.25), 2)


def day_means(fit: pd.DataFrame, clip_s: float = 1800.0, min_rows: int = 100) -> pd.DataFrame:
    """Airport-day mean residual of the fitting rows within clip_s of the prediction, with its sampling variance."""
    o = fit[fit["r"].abs() <= clip_s]
    day = o.groupby(["airport", "bday"])["r"].agg(["mean", "var", "size"])
    day = day[day["size"] >= min_rows]
    return pd.DataFrame({"airport": day.index.get_level_values("airport"), "mean": day["mean"].to_numpy(), "sampling_var": (day["var"] / day["size"]).to_numpy()})


def fit_mixture(days: pd.DataFrame, tau: dict) -> dict:
    """Maximum-likelihood (pi, k) of the pooled scale mixture on standardised airport-day means (centred per airport)."""
    d = days[days["airport"].isin(list(tau))].copy()
    d["mean"] = d["mean"] - d.groupby("airport")["mean"].transform("mean")
    t = d["airport"].map(tau).to_numpy(dtype=float)
    z, v = d["mean"].to_numpy() / t, d["sampling_var"].to_numpy() / t ** 2

    def loglik(pi: float, k: float) -> float:
        s1, s2 = 1.0 + v, k ** 2 + v
        dens = (1 - pi) * np.exp(-0.5 * z ** 2 / s1) / np.sqrt(2 * np.pi * s1) + pi * np.exp(-0.5 * z ** 2 / s2) / np.sqrt(2 * np.pi * s2)
        return float(np.log(dens + 1e-300).sum())

    base = loglik(0.0, 1.0)
    best = max(((loglik(p, k), p, k) for p in PI_GRID for k in K_GRID), key=lambda x: x[0])
    return {"pi": float(best[1]), "k": float(best[2]), "loglik": best[0], "loglik_gaussian": base, "days": int(len(z)),
            "share_abs_z_over_3": float((np.abs(z) > 3).mean())}


def solve_day_mix(p: Prepared, total_s: float, noise_sd_s: float, noise_wide_share: float, tau_s: float, pi: float, k: float,
                  settings: Settings = Settings(), grid_points: int | None = None) -> tuple[np.ndarray, dict]:
    if settings.heavy_updates or settings.heavy_tilt:
        raise ValueError("solve_day_mix implements v48's heavy-row handling only (heavy rows keep their prediction).")
    n = len(p.yhat)
    delta = (float(total_s) - p.reference_total) / GRID_S
    info = {"rows": n, "delta_s": float(total_s) - p.reference_total, "heavy": int(len(p.heavy)), "shift_s": 0.0, "score": 0.0, "n_eff": p.n_eff, "used": False}
    if not np.isfinite(delta) or abs(delta) >= N // 2 - 1:
        return p.yhat.copy(), info
    share = float(noise_wide_share)
    noise = (1 - share) * _gauss_cf(noise_sd_s) + sum(0.5 * share * _gauss_cf(s) for s in settings.noise_wide_sd_s)
    everything = p.bulk_cf * noise
    for j in range(len(p.heavy)):
        everything = everything * p.heavy_cf[j]
    floor = 1e-300
    wide = k * tau_s
    shift = 0.0
    if tau_s > 0:
        q_rest = np.maximum(np.fft.irfft(everything, n=N), 0.0)
        span = 4 * max(tau_s, wide if pi > 0 else tau_s)
        points = grid_points or (settings.shift_grid if pi == 0 else max(settings.shift_grid, int(np.ceil(settings.shift_grid * span / (4 * tau_s))) | 1))
        grid = np.linspace(-span, span, points)
        prior = (1 - pi) * np.exp(-0.5 * (grid / tau_s) ** 2) / tau_s + (pi * np.exp(-0.5 * (grid / wide) ** 2) / wide if pi > 0 else 0.0)
        w = prior * (_at(q_rest, delta + p.n_eff * grid / GRID_S) + floor)
        shift = float(np.sum(w * grid) / np.sum(w))
    shift_cf = (1 - pi) * _gauss_cf(p.n_eff * tau_s) + (pi * _gauss_cf(p.n_eff * wide) if pi > 0 else 0.0)
    q_total = np.maximum(np.fft.irfft(everything * shift_cf, n=N), 0.0)
    h = settings.score_halfwidth_bins
    score = float(np.log(_at(q_total, np.array([delta + h]))[0] + floor) - np.log(_at(q_total, np.array([delta - h]))[0] + floor)) / (2 * h * GRID_S)
    post = p.prior.copy()
    heavy = set(int(r) for r in p.heavy)
    for rows, y, e, w0 in p.store:
        y, e, w0 = y.astype(float), e.astype(float), w0.astype(float)
        w = w0 * np.exp(np.clip(-score * e, -settings.score_clip, settings.score_clip))
        post[rows] = (w * y).sum(axis=1) / w.sum(axis=1)
        for r in rows:
            if int(r) in heavy:
                post[r] = p.prior[r]
    new = p.yhat + (post - p.prior) + np.where(p.is_heavy, 0.0, shift)
    info.update({"shift_s": shift, "score": score, "used": True})
    return new, info
