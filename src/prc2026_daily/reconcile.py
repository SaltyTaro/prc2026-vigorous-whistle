"""Posterior mean of each departure's taxi time given the airport-day's published total of positive off-block delay.

Per departure i: a_i = takeoff - schedule (known), prediction yhat_i, label y_i (hidden), positive delay pos_i = max(a_i - y_i, 0).
Published: S = sum_i pos_i + noise, per airport and off-block day.

Row model. Each row draws a scenario from the recorded outcomes of comparable rows (its cell) in the fitting months:
    ordinary     |y_j - yhat_j| <= 600 s, or a label above the prediction with off-block after schedule: y = yhat_i + (y_j - yhat_j)
    fill         off-block = schedule within 120 s (schedule interval above 1,800 s):                    y = a_i + (y_j - a_j)
    before       off-block before schedule with a large error (one-day rows, early clocks):             y = y_j where that stays before
                                                                                                         schedule, else a_i + (y_j - a_j)
    lower        a label far below the prediction, off-block after schedule:                            y = min(y_j, yhat_i)
Rows whose scenarios spread by more than `tilt_sd_s` get scenario weights proportional to exp(lambda y), lambda chosen so that the
weighted mean of the scenarios equals the row's prediction: the stack's own hedge stays the prior, the cell gives its shape.

Day model. Rows are independent given a day shift b ~ N(0, tau^2) of the ordinary labels; the total's deviation from the prediction's
own total is sum_i e_i - n_eff b + noise with e_i = pos_i - max(a_i - yhat_i, 0). The noise is a Gaussian core plus two wide Gaussians.

Posterior by convolution on a grid: rows with a wide e_i ("heavy") get the exact leave-one-out posterior over their scenarios, the rest
the first-order tilt by the score of the total's density, and every other row the posterior mean of b. The update applied to a
prediction is posterior mean minus prior mean of its scenarios, so a row the total says nothing about keeps its prediction.
No label of the month read enters; the cells come from other months.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID_S = 30.0
N = 1 << 14
SMALL_S, FILL_S, FILL_MIN_A = 600.0, 120.0, 1800.0


@dataclass(frozen=True)
class Settings:
    heavy_sd_s: float = 900.0        # a row is heavy when the sd of its e_i exceeds this
    max_heavy: int = 64
    heavy_updates: bool = True       # False: heavy rows enter the convolution but keep their prediction
    tilt_sd_s: float = 300.0         # rows whose scenarios spread more than this get prior weights matched to the prediction
    noise_wide_sd_s: tuple = (3600.0, 28800.0)
    score_clip: float = 3.0
    score_halfwidth_bins: int = 10
    shift_grid: int = 81


def scenarios(a: np.ndarray, yhat: np.ndarray, sample_a: np.ndarray, sample_yhat: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    """Scenario labels, rows x samples."""
    a, yhat = np.asarray(a, dtype=float)[:, None], np.asarray(yhat, dtype=float)[:, None]
    sa, sy, sl = (np.asarray(v, dtype=float)[None, :] for v in (sample_a, sample_yhat, sample_y))
    r = sl - sy
    small = np.abs(r) <= SMALL_S
    fill = ~small & (np.abs(sl - sa) <= FILL_S) & (sa > FILL_MIN_A)
    before = ~small & ~fill & (sl > sa + FILL_S)
    lower = ~small & ~fill & ~before & (r < 0)
    relative = a + (sl - sa)
    out = yhat + r                                                   # ordinary, and labels above the prediction after schedule
    out = np.where(fill, relative, out)
    out = np.where(before, np.where(sl > a, sl, relative), out)
    return np.where(lower, np.minimum(sl, yhat), out)


def matched_weights(y: np.ndarray, target: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Row weights proportional to exp(lambda y) whose weighted mean of y equals target (where target lies inside the scenarios)."""
    scale = y.std(axis=1, keepdims=True) + 1e-9
    z = (y - np.asarray(target, dtype=float)[:, None]) / scale
    lo, hi = np.full(len(y), -80.0), np.full(len(y), 80.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        x = mid[:, None] * z
        w = np.exp(x - x.max(axis=1, keepdims=True))
        above = (w * z).sum(axis=1) > 0
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    x = (0.5 * (lo + hi))[:, None] * z
    w = np.exp(x - x.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def _pmf(e: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Row pmfs of e (rows x samples, seconds) on the wrapped grid, mass split linearly between neighbouring bins."""
    rows = e.shape[0]
    x = e / GRID_S
    lo = np.floor(x)
    frac = x - lo
    base = (np.arange(rows)[:, None] * N)
    lo = lo.astype(np.int64)
    flat = np.concatenate([(base + np.mod(lo, N)).ravel(), (base + np.mod(lo + 1, N)).ravel()])
    mass = np.concatenate([((1 - frac) * weights).ravel(), (frac * weights).ravel()])
    return np.bincount(flat, weights=mass, minlength=rows * N).reshape(rows, N)


def _gauss_cf(sd_s: float) -> np.ndarray:
    w = 2 * np.pi * np.arange(N // 2 + 1) / N
    return np.exp(-0.5 * (w * sd_s / GRID_S) ** 2)


def _at(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Density q (wrapped grid) at fractional bin positions x, linear interpolation."""
    lo = np.floor(x)
    frac = x - lo
    lo = lo.astype(np.int64)
    return (1 - frac) * q[np.mod(lo, N)] + frac * q[np.mod(lo + 1, N)]


@dataclass
class Prepared:
    """One airport-day after the row model: everything the posterior needs except the published total and the shift prior."""
    yhat: np.ndarray
    reference_total: float
    prior: np.ndarray
    is_heavy: np.ndarray
    heavy: np.ndarray
    heavy_cf: np.ndarray
    bulk_cf: np.ndarray
    store: list
    n_eff: float
    mean_e: float        # prior mean of the sum of e_i, all rows
    var_bulk: float      # prior variance of the sum of e_i over the rows that are not heavy


def prepare_day(a: np.ndarray, yhat: np.ndarray, groups: list, settings: Settings = Settings()) -> Prepared:
    """groups: [(row positions, (sample_a, sample_yhat, sample_y))], a partition of the day's rows by cell."""
    a, yhat = np.asarray(a, dtype=float), np.asarray(yhat, dtype=float)
    n = len(a)
    reference = np.maximum(a - yhat, 0.0)
    prior, sd, late, mean = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    cf = np.empty((n, N // 2 + 1), dtype=np.complex128)
    store = []
    for rows, (sa, sy, sl) in groups:
        y = scenarios(a[rows], yhat[rows], sa, sy, sl)
        w = np.full(y.shape, 1.0 / y.shape[1])
        wide = y.std(axis=1) > settings.tilt_sd_s
        if wide.any():
            w[wide] = matched_weights(y[wide], yhat[rows][wide])
        pos = np.maximum(a[rows][:, None] - y, 0.0)
        e = pos - reference[rows][:, None]
        mean[rows] = (w * e).sum(axis=1)
        prior[rows], late[rows] = (w * y).sum(axis=1), (w * (pos > 0)).sum(axis=1)
        sd[rows] = np.sqrt(np.maximum((w * (e - mean[rows][:, None]) ** 2).sum(axis=1), 0.0))
        cf[rows] = np.fft.rfft(_pmf(e, w), axis=1)
        store.append((rows, y.astype(np.float32), e.astype(np.float32), w.astype(np.float32)))
    heavy = np.flatnonzero(sd > settings.heavy_sd_s)
    if len(heavy) > settings.max_heavy:
        heavy = heavy[np.argsort(-sd[heavy])[:settings.max_heavy]]
    is_heavy = np.zeros(n, dtype=bool)
    is_heavy[heavy] = True
    return Prepared(yhat=yhat, reference_total=float(reference.sum()), prior=prior, is_heavy=is_heavy, heavy=heavy, heavy_cf=cf[heavy].copy(), bulk_cf=np.prod(cf[~is_heavy], axis=0),
                    store=store, n_eff=float(late[~is_heavy].sum()), mean_e=float(mean.sum()), var_bulk=float((sd[~is_heavy] ** 2).sum()))


def shift_estimate(p: Prepared, total_s: float) -> float:
    """Label-free estimate of the day's shift: the ordinary labels' common offset that would close the total, seconds."""
    return -(float(total_s) - p.reference_total - p.mean_e) / max(p.n_eff, 1.0)


def solve_day(p: Prepared, total_s: float, noise_sd_s: float, noise_wide_share: float, shift_mean_s: float, shift_sd_s: float,
              settings: Settings = Settings()) -> tuple[np.ndarray, dict]:
    """Reconciled predictions of a prepared airport-day; the day shift has the prior N(shift_mean_s, shift_sd_s^2)."""
    n = len(p.yhat)
    delta = (float(total_s) - p.reference_total + p.n_eff * shift_mean_s) / GRID_S   # the total with the prior mean of the shift taken out
    info = {"rows": n, "delta_s": float(total_s) - p.reference_total, "heavy": int(len(p.heavy)), "shift_s": 0.0, "score": 0.0, "n_eff": p.n_eff, "used": False}
    if not np.isfinite(delta) or abs(delta) >= N // 2 - 1:
        return p.yhat.copy(), info
    share = float(noise_wide_share)
    noise = (1 - share) * _gauss_cf(noise_sd_s) + sum(0.5 * share * _gauss_cf(s) for s in settings.noise_wide_sd_s)
    base = p.bulk_cf * noise
    shift_cf = _gauss_cf(p.n_eff * shift_sd_s)
    k = len(p.heavy)
    prefix = np.ones((k + 1, N // 2 + 1), dtype=np.complex128)
    suffix = np.ones((k + 1, N // 2 + 1), dtype=np.complex128)
    for j in range(k):
        prefix[j + 1] = prefix[j] * p.heavy_cf[j]
        suffix[k - 1 - j] = suffix[k - j] * p.heavy_cf[k - 1 - j]
    everything = base * prefix[k]
    floor = 1e-300

    shift = float(shift_mean_s)
    if shift_sd_s > 0:  # day shift b = mean + b2: the sum's component is -n_eff b2
        q_rest = np.maximum(np.fft.irfft(everything, n=N), 0.0)
        grid = np.linspace(-4 * shift_sd_s, 4 * shift_sd_s, settings.shift_grid)
        w = np.exp(-0.5 * (grid / shift_sd_s) ** 2) * (_at(q_rest, delta + p.n_eff * grid / GRID_S) + floor)
        shift += float(np.sum(w * grid) / np.sum(w))

    q_total = np.maximum(np.fft.irfft(everything * shift_cf, n=N), 0.0)
    h = settings.score_halfwidth_bins
    score = float(np.log(_at(q_total, np.array([delta + h]))[0] + floor) - np.log(_at(q_total, np.array([delta - h]))[0] + floor)) / (2 * h * GRID_S)

    post = p.prior.copy()
    position = {int(r): j for j, r in enumerate(p.heavy)}
    for rows, y, e, w0 in p.store:
        y, e, w0 = y.astype(float), e.astype(float), w0.astype(float)
        w = w0 * np.exp(np.clip(-score * e, -settings.score_clip, settings.score_clip))
        post[rows] = (w * y).sum(axis=1) / w.sum(axis=1)
        for local, r in enumerate(rows):
            if int(r) in position:
                post[r] = p.prior[r]
                if settings.heavy_updates:
                    j = position[int(r)]
                    q = np.maximum(np.fft.irfft(base * shift_cf * prefix[j] * suffix[j + 1], n=N), 0.0)
                    w = w0[local] * (_at(q, delta - e[local] / GRID_S) + floor)
                    post[r] = float(np.sum(w * y[local]) / np.sum(w))
    new = p.yhat + (post - p.prior) + np.where(p.is_heavy, 0.0, shift)
    info.update({"shift_s": shift, "score": score, "used": True})
    return new, info


def reconcile_day(a: np.ndarray, yhat: np.ndarray, groups: list, total_s: float, noise_sd_s: float, noise_wide_share: float, tau_s: float,
                  settings: Settings = Settings()) -> tuple[np.ndarray, dict]:
    """Reconciled predictions of one airport-day with the fixed shift prior N(0, tau_s^2)."""
    return solve_day(prepare_day(a, yhat, groups, settings), total_s, noise_sd_s, noise_wide_share, 0.0, tau_s, settings)


def month_prior(estimates: np.ndarray, variances: np.ndarray, floor_sd_s: float = 3.0) -> tuple[float, float]:
    """Label-free shift prior of an airport-month from its days' shift estimates and their model variances.

    mean: the median of the estimates, shrunk by max(0, 1 - se^2 / median^2) with se = 1.2533 s / sqrt(days);
    sd:   sqrt(max(s^2 - median model variance, floor^2)), s = the estimates' robust sd (1.4826 MAD).
    """
    x = np.asarray(estimates, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 0.0, float(floor_sd_s)
    centre = float(np.median(x))
    s = 1.4826 * float(np.median(np.abs(x - centre)))
    se = 1.2533 * s / np.sqrt(len(x))
    mean = centre * max(0.0, 1.0 - se ** 2 / centre ** 2) if centre != 0 else 0.0
    return mean, float(np.sqrt(max(s ** 2 - float(np.median(variances)), floor_sd_s ** 2)))


__all__ = ["GRID_S", "N", "Settings", "Prepared", "scenarios", "matched_weights", "prepare_day", "shift_estimate", "solve_day", "reconcile_day", "month_prior"]
