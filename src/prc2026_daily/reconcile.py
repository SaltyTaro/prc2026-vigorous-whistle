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
With components (prepare_day's `component`, solve_day's `shift_cov`) each component k of the rows has its own shift b_k, jointly
N(mean, cov): the total sees s = sum_k n_k (b_k - mean_k) only, and E[b_k | total] = mean_k + (cov n)_k / (n' cov n) E[s | total].

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
    heavy_temper: float = 1.0        # heavy_updates: the leave-one-out likelihood enters to this power (1 = the exact posterior)
    heavy_tilt: bool = False         # without heavy_updates: heavy rows get the first-order tilt of the other rows (no day shift)
    heavy_prior: str = "tilt"        # "groups": heavy rows get their prior weights from group_weights; "bounded": the same, and
                                     # their outcomes without a visible delay count as the schedule interval, so that the total
                                     # moves a row between its delay outcomes and its schedule interval and never beyond
    pin_s: float = 0.0               # heavy rows predicted within this of their schedule interval (declared fills) keep their prediction
    heavy_keep_clock: bool = False   # heavy rows that hold an NM clock keep their prediction (prepare_day's `keep`)
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


def matched_weights(y: np.ndarray, target: np.ndarray, iterations: int = 50, allowed: np.ndarray | None = None) -> np.ndarray:
    """Row weights proportional to exp(lambda y) whose weighted mean of y equals target (where target lies inside the scenarios).
    allowed (rows x samples): the scenarios that may carry weight; every row needs one."""
    scale = y.std(axis=1, keepdims=True) + 1e-9
    z = (y - np.asarray(target, dtype=float)[:, None]) / scale
    lo, hi = np.full(len(y), -80.0), np.full(len(y), 80.0)
    if allowed is None:
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            x = mid[:, None] * z
            w = np.exp(x - x.max(axis=1, keepdims=True))
            above = (w * z).sum(axis=1) > 0
            hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
        x = (0.5 * (lo + hi))[:, None] * z
        w = np.exp(x - x.max(axis=1, keepdims=True))
        return w / w.sum(axis=1, keepdims=True)
    allowed = np.asarray(allowed, dtype=bool)
    if not allowed.any(axis=1).all():
        raise ValueError("A row has no allowed scenario.")

    def weights(slope: np.ndarray) -> np.ndarray:
        x = np.where(allowed, slope[:, None] * z, -np.inf)
        return np.exp(x - x.max(axis=1, keepdims=True))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        w = weights(mid)
        above = (w * z).sum(axis=1) > 0
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    w = weights(0.5 * (lo + hi))
    return w / w.sum(axis=1, keepdims=True)


def group_weights(y: np.ndarray, a: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Prior weights of rows hedged between a delay and none (rows x samples).

    The published total tells a visible delay (label below the schedule interval) from none (a fill, an off-block before schedule,
    a one-day label); it cannot tell the outcomes without delay apart. The scenarios of each of the two groups keep the proportions
    their cell recorded; the groups' weights make the mean the target where it lies between the two groups' means. A target above the
    mean without delay is matched by a tilt inside that group alone, a target below the mean with delay by a tilt inside the other:
    such a row holds one hypothesis and the total does not move it between the two."""
    y, target = np.asarray(y, dtype=float), np.asarray(target, dtype=float)
    high = y >= np.asarray(a, dtype=float)[:, None] - FILL_S
    n_high = high.sum(axis=1)
    n_low = y.shape[1] - n_high
    both = (n_high > 0) & (n_low > 0)
    m_high = (y * high).sum(axis=1) / np.maximum(n_high, 1)
    m_low = (y * ~high).sum(axis=1) / np.maximum(n_low, 1)
    between = both & (target >= m_low) & (target <= m_high)
    out = np.empty_like(y)
    if between.any():
        share = ((target - m_low) / np.maximum(m_high - m_low, 1e-9))[between][:, None]
        out[between] = share * high[between] / n_high[between][:, None] + (1 - share) * ~high[between] / n_low[between][:, None]
    single = ~both
    if single.any():
        out[single] = matched_weights(y[single], target[single])
    above, below = both & (target > m_high), both & (target < m_low)
    if above.any():
        out[above] = matched_weights(y[above], target[above], allowed=high[above])
    if below.any():
        out[below] = matched_weights(y[below], target[below], allowed=~high[below])
    return out


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
    component: np.ndarray | None = None   # shift component of each row
    n_eff_k: np.ndarray | None = None     # n_eff by component
    pinned: np.ndarray | None = None      # rows that keep their prediction when heavy: Settings.pin_s, and prepare_day's `keep`


def prepare_day(a: np.ndarray, yhat: np.ndarray, groups: list, settings: Settings = Settings(), component: np.ndarray | None = None, components: int = 1,
                keep: np.ndarray | None = None) -> Prepared:
    """groups: [(row positions, (sample_a, sample_yhat, sample_y))], a partition of the day's rows by cell.
    component: the shift component of each row (0 ... components - 1); one component when absent.
    keep: rows that keep their prediction when they are heavy (they still enter the day's arithmetic)."""
    a, yhat = np.asarray(a, dtype=float), np.asarray(yhat, dtype=float)
    n = len(a)
    component = np.zeros(n, dtype=np.int64) if component is None else np.asarray(component, dtype=np.int64)
    if len(component) != n or (n and (component.min() < 0 or component.max() >= components)):
        raise ValueError("Every row needs a shift component below `components`.")
    reference = np.maximum(a - yhat, 0.0)
    prior, sd, late, mean = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    if settings.heavy_prior not in ("tilt", "groups", "bounded"):
        raise ValueError("heavy_prior: tilt, groups or bounded.")
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
        if settings.heavy_prior != "tilt":   # the rows that are heavy by the weights above stay the heavy rows
            k = sd[rows] > settings.heavy_sd_s
            if k.any():
                if settings.heavy_prior == "bounded":
                    y[k] = np.where(y[k] >= a[rows][k][:, None] - FILL_S, a[rows][k][:, None], y[k])
                w[k] = group_weights(y[k], a[rows][k], yhat[rows][k])
                mean[rows[k]] = (w[k] * e[k]).sum(axis=1)
                prior[rows[k]], late[rows[k]] = (w[k] * y[k]).sum(axis=1), (w[k] * (pos[k] > 0)).sum(axis=1)
        cf[rows] = np.fft.rfft(_pmf(e, w), axis=1)
        store.append((rows, y.astype(np.float32), e.astype(np.float32), w.astype(np.float32)))
    heavy = np.flatnonzero(sd > settings.heavy_sd_s)
    if len(heavy) > settings.max_heavy:
        heavy = heavy[np.argsort(-sd[heavy])[:settings.max_heavy]]
    is_heavy = np.zeros(n, dtype=bool)
    is_heavy[heavy] = True
    return Prepared(yhat=yhat, reference_total=float(reference.sum()), prior=prior, is_heavy=is_heavy, heavy=heavy, heavy_cf=cf[heavy].copy(), bulk_cf=np.prod(cf[~is_heavy], axis=0),
                    store=store, n_eff=float(late[~is_heavy].sum()), mean_e=float(mean.sum()), var_bulk=float((sd[~is_heavy] ** 2).sum()),
                    component=component, n_eff_k=np.bincount(component[~is_heavy], weights=late[~is_heavy], minlength=components).astype(float),
                    pinned=((np.abs(yhat - a) <= settings.pin_s) if settings.pin_s > 0 else np.zeros(n, dtype=bool)) | (np.zeros(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)))


def shift_estimate(p: Prepared, total_s: float) -> float:
    """Label-free estimate of the day's shift: the ordinary labels' common offset that would close the total, seconds."""
    return -(float(total_s) - p.reference_total - p.mean_e) / max(p.n_eff, 1.0)


def solve_day(p: Prepared, total_s: float, noise_sd_s: float, noise_wide_share: float, shift_mean_s, shift_sd_s: float,
              settings: Settings = Settings(), shift_cov: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Reconciled predictions of a prepared airport-day; the day shift has the prior N(shift_mean_s, shift_sd_s^2).
    With shift_cov (components x components, s^2) the components' shifts have the prior N(shift_mean_s, shift_cov) and shift_sd_s is
    not read; info["shift_s"] is then the mean shift over the day's late rows and info["shifts_s"] holds the components'."""
    n = len(p.yhat)
    if shift_cov is None:
        offset, sum_sd = p.n_eff * shift_mean_s, p.n_eff * shift_sd_s
    else:
        shift_cov = np.asarray(shift_cov, dtype=float)
        means = np.broadcast_to(np.asarray(shift_mean_s, dtype=float), p.n_eff_k.shape).astype(float)
        if shift_cov.shape != (len(p.n_eff_k), len(p.n_eff_k)):
            raise ValueError("shift_cov does not match the day's components.")
        load = shift_cov @ p.n_eff_k                                  # cov(b_k, s)
        offset, sum_sd = float(p.n_eff_k @ means), float(np.sqrt(max(float(p.n_eff_k @ load), 0.0)))
    delta = (float(total_s) - p.reference_total + offset) / GRID_S   # the total with the prior mean of the shift taken out
    info = {"rows": n, "delta_s": float(total_s) - p.reference_total, "heavy": int(len(p.heavy)), "shift_s": 0.0, "score": 0.0, "n_eff": p.n_eff, "used": False}
    if not np.isfinite(delta) or abs(delta) >= N // 2 - 1:
        return p.yhat.copy(), info
    share = float(noise_wide_share)
    noise = (1 - share) * _gauss_cf(noise_sd_s) + sum(0.5 * share * _gauss_cf(s) for s in settings.noise_wide_sd_s)
    base = p.bulk_cf * noise
    shift_cf = _gauss_cf(sum_sd)
    k = len(p.heavy)
    prefix = np.ones((k + 1, N // 2 + 1), dtype=np.complex128)
    suffix = np.ones((k + 1, N // 2 + 1), dtype=np.complex128)
    for j in range(k):
        prefix[j + 1] = prefix[j] * p.heavy_cf[j]
        suffix[k - 1 - j] = suffix[k - j] * p.heavy_cf[k - 1 - j]
    everything = base * prefix[k]
    floor = 1e-300

    if shift_cov is None:
        shift = float(shift_mean_s)
        if shift_sd_s > 0:  # day shift b = mean + b2: the sum's component is -n_eff b2
            q_rest = np.maximum(np.fft.irfft(everything, n=N), 0.0)
            grid = np.linspace(-4 * shift_sd_s, 4 * shift_sd_s, settings.shift_grid)
            w = np.exp(-0.5 * (grid / shift_sd_s) ** 2) * (_at(q_rest, delta + p.n_eff * grid / GRID_S) + floor)
            shift += float(np.sum(w * grid) / np.sum(w))
        row_shift = shift
    else:
        shifts = means.copy()
        if sum_sd > 0:      # s = sum_k n_k (b_k - mean_k): the sum's component is -s; b_k - mean_k = load_k / var(s) s + a part the total does not see
            q_rest = np.maximum(np.fft.irfft(everything, n=N), 0.0)
            grid = np.linspace(-4 * sum_sd, 4 * sum_sd, settings.shift_grid)
            w = np.exp(-0.5 * (grid / sum_sd) ** 2) * (_at(q_rest, delta + grid / GRID_S) + floor)
            shifts = shifts + load / sum_sd ** 2 * float(np.sum(w * grid) / np.sum(w))
        row_shift = shifts[p.component]
        shift = float(p.n_eff_k @ shifts / p.n_eff) if p.n_eff > 0 else float(shifts.mean())
        info["shifts_s"] = [float(v) for v in shifts]

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
                if not settings.heavy_updates and settings.heavy_tilt:
                    continue
                post[r] = p.prior[r]
                if settings.heavy_updates and not p.pinned[r]:
                    j = position[int(r)]
                    q = np.maximum(np.fft.irfft(base * shift_cf * prefix[j] * suffix[j + 1], n=N), 0.0)
                    likelihood = _at(q, delta - e[local] / GRID_S) + floor
                    if settings.heavy_temper != 1.0:
                        likelihood = (likelihood / likelihood.max()) ** settings.heavy_temper
                    w = w0[local] * likelihood
                    post[r] = float(np.sum(w * y[local]) / np.sum(w))
    new = p.yhat + (post - p.prior) + np.where(p.is_heavy, 0.0, row_shift)
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
