"""The published daily departure counts as a constraint on each departure's off-block day (configs/scout_count_days_v1.json).

EUROCONTROL's daily figure FLT_DEP_IFR_2 counts the departures by off-block day. With the off-block time rounded to the minute it
equals our rows' count by recorded off-block day on every 2025 day at LIRF, EGLL and LEBL, and on all but one or two days at LEMD and
LTFM (month-end days aside: departures that take off in the next month are not in the month's rows). A prediction fixes a row's
off-block day (takeoff - prediction); where the predicted days disagree with the published counts, some rows sit on the wrong side of a
midnight. This layer finds which, by v48's own row model and the published totals:

  row model   v48's scenarios (src/prc2026_daily/reconcile.scenarios, the cells of the fitting months) with the prior weights matched
              to the row's value (matched_weights, as prepare_day); each scenario's off-block day = round(takeoff - label, 1 min)
  candidates  rows whose scenarios put at least MASS_MIN on each of two adjacent days (with mass on more days: the two with the most,
              if adjacent; otherwise the row is skipped); the rest are certain on their value's day.
              A candidate is heavy when its two days differ by more than HEAVY_POS_S in expected positive delay (max(a - y, 0)): the
              day's published delay total can tell them apart; the others are light
  likelihood  per complete day (every day of the month but the last): the published count against the certain rows plus the
              candidates placed there, exact with probability 1 - eps, off by one with eps / 2 each (eps from the airport's 2025
              days); per day with heavy candidates placed on it, the density of the published delay total (v48's day model: the
              day's other rows by prepare_day, the heavy rows' positive delay under their restricted scenarios, the figure's noise
              and the day shift N(0, tau^2))
  posterior   exact, by forward-backward over the month's midnights (state: the number of light candidates on the earlier day and
              the heavy candidates' days); light candidates' marginals by leave-one-out Poisson-binomial
  flagged     a midnight whose earlier or later day's published count differs from the count of the values' days; candidates of
              other midnights keep their values (where the counts agree, the update would rest on the row model's split of a value)
  big         a candidate whose two days differ by more than HEAVY_POS_S in mean label moves only when the posterior puts it on the
              other side of midnight from its value's day (more than half); otherwise the update would rest on the row model's split
  value       each candidate of a flagged midnight: its value + (P(earlier day | counts, totals) - P(earlier day)) x (mean of its scenarios on the earlier day -
              mean on the later day), the change of its posterior mean (a candidate the counts say nothing about keeps its value);
              every other row keeps its value; floor -12 s
No label of a month read enters: labels enter only the cells (fitting months) and the airports' 2025 count exactness.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prc2026_daily.reconcile import GRID_S, N, Settings, _at, _gauss_cf, _pmf, matched_weights, prepare_day, scenarios

EXACT_MIN_SHARE, MASS_MIN, HEAVY_POS_S, MAX_HEAVY, CHUNK, FLOOR_S = 0.99, 1e-3, 3600.0, 4, 3000, -12.0


def minute_day(t: pd.Series) -> np.ndarray:
    return pd.to_datetime(t, utc=True).dt.round("min").dt.strftime("%Y-%m-%d").to_numpy()


def count_exactness(dep: pd.DataFrame, published: pd.Series) -> pd.DataFrame:
    """Per airport over 2025: the share of non-month-end days whose published count equals the in-figure rows counted by minute-rounded
    recorded off-block day, and eps = (days off + 1) / (days + 2). dep: airport, block (recorded off-block time), in_figure."""
    c = dep[dep["in_figure"]].assign(day=lambda x: minute_day(x["block"])).groupby(["airport", "day"]).size().rename("n")
    x = pd.DataFrame({"n_pub": published}).join(c, how="inner")
    day = pd.to_datetime(x.index.get_level_values("day"))
    x = x[((day + pd.Timedelta(days=1)).day != 1) & (day.year == 2025)]
    off = (x["n_pub"] != x["n"]).groupby(level=0)
    out = pd.DataFrame({"days": off.size(), "days_off": off.sum()})
    out["share"] = 1 - out["days_off"] / out["days"]
    out["eps"] = (out["days_off"] + 1) / (out["days"] + 2)
    out["exact"] = out["share"] >= EXACT_MIN_SHARE
    return out


@dataclass
class Candidate:
    position: int          # row position in the airport-month frame
    k: int                 # index of the later day in the month's day list (the midnight between day k-1 and day k)
    p_early: float         # prior P(off-block on day k-1)
    mean: tuple            # (mean label if day k-1, mean label if day k)
    heavy: bool
    big: bool = False      # its two days differ by more than HEAVY_POS_S in mean label
    pos_e: tuple = ()      # heavy: ((pos values, weights) if day k-1, (pos values, weights) if day k)


def row_days(f: pd.DataFrame, groups: list, days: list[str], settings: Settings = Settings()) -> tuple[np.ndarray, list[Candidate], int]:
    """Day of each row's value (index into days, -1 outside), the candidates, and the number of rows spread over more than two days or
    over two days that are not adjacent (kept on their value's day). f: takeoff, a, value, positionally aligned with groups."""
    index = {d: i for i, d in enumerate(days)}
    takeoff = pd.to_datetime(f["takeoff"], utc=True).reset_index(drop=True)
    value = f["value"].to_numpy(dtype=float)
    a = f["a"].to_numpy(dtype=float)
    base_day = np.array([index.get(d, -1) for d in minute_day(takeoff - pd.to_timedelta(value, unit="s"))])
    out, skipped = [], 0
    t_ns = takeoff.dt.as_unit("ns").astype("int64").to_numpy()
    for rows, (sa, sy, sl) in groups:
        for start in range(0, len(rows), CHUNK):
            r = rows[start:start + CHUNK]
            y = scenarios(a[r], value[r], sa, sy, sl)
            w = np.full(y.shape, 1.0 / y.shape[1])
            wide = y.std(axis=1) > settings.tilt_sd_s
            if wide.any():
                w[wide] = matched_weights(y[wide], value[r][wide])
            block_ns = t_ns[r][:, None] - (y * 1e9).astype(np.int64)
            minute = np.round(block_ns / 6e10) * 6e10
            day_no = np.floor(minute / 8.64e13).astype(np.int64)          # days since the epoch
            lo, hi = day_no.min(axis=1), day_no.max(axis=1)
            for j in np.flatnonzero(lo != hi):
                dn = day_no[j]
                mass = pd.Series(w[j]).groupby(dn).sum()
                mass = mass[mass >= MASS_MIN]
                if len(mass) < 2:
                    continue
                if len(mass) > 2:   # the two days with the most mass, if adjacent; the rest of its mass is dropped
                    mass = mass.sort_values(ascending=False).iloc[:2].sort_index()
                if mass.index[1] - mass.index[0] != 1:
                    skipped += 1
                    continue
                early_no, late_no = int(mass.index[0]), int(mass.index[1])
                late_day = (pd.Timestamp(0, tz="UTC") + pd.Timedelta(days=late_no)).strftime("%Y-%m-%d")
                if late_day not in index:
                    continue
                e, l = dn == early_no, dn == late_no
                p_early = float(mass.iloc[0] / mass.sum())
                me = float(np.sum(w[j][e] * y[j][e]) / np.sum(w[j][e]))
                ml = float(np.sum(w[j][l] * y[j][l]) / np.sum(w[j][l]))
                pos = np.maximum(a[r[j]] - y[j], 0.0)
                pe, pl = float(np.sum(w[j][e] * pos[e]) / np.sum(w[j][e])), float(np.sum(w[j][l] * pos[l]) / np.sum(w[j][l]))
                heavy = abs(pe - pl) > HEAVY_POS_S
                pos_e = ((pos[e], w[j][e] / w[j][e].sum()), (pos[l], w[j][l] / w[j][l].sum())) if heavy else ()
                out.append(Candidate(int(r[j]), index[late_day], p_early, (me, ml), heavy, abs(me - ml) > HEAVY_POS_S, pos_e))
    return base_day, out, skipped


def _poisson_binomial(p: np.ndarray) -> np.ndarray:
    q = np.array([1.0])
    for x in p:
        q = np.convolve(q, [1 - x, x])
    return q


def _log_density(cf_rest: np.ndarray, extra: list, delta_bins: float) -> float:
    """log density of the day's total (on the grid, relative) with the heavy candidates' positive delays added."""
    cf = cf_rest.copy()
    for values, weights in extra:
        cf = cf * np.fft.rfft(_pmf(values[None, :], weights[None, :]), axis=1)[0]
    q = np.maximum(np.fft.irfft(cf, n=N), 0.0)
    return float(np.log(_at(q, np.array([delta_bins]))[0] + 1e-300))


def solve_month(f: pd.DataFrame, groups: list, days: list[str], n_pub: dict, complete: set, eps: float, noise_sd_s: float, noise_wide_share: float,
                tau_s: float, resolve, pub_total: dict, settings: Settings = Settings()) -> tuple[np.ndarray, dict]:
    """New values of one airport-month (positional, as f) and a report. f: takeoff, a, value (+ the cell keys used by groups).
    n_pub: published count by day; complete: the days whose count is read; resolve: the cell partition of a frame of rows (Cells.resolve);
    pub_total: published delay total (DLY_ALL_PRE_2 x 60, seconds) by day."""
    base_day, cands, skipped = row_days(f, groups, days, settings)
    value = f["value"].to_numpy(dtype=float)
    a = f["a"].to_numpy(dtype=float)
    K = len(days)
    cand_pos = {c.position for c in cands}
    certain = np.bincount(base_day[(base_day >= 0) & ~np.isin(np.arange(len(f)), list(cand_pos))], minlength=K).astype(int)
    by_k = {k: [c for c in cands if c.k == k] for k in range(K + 1)}
    for k in by_k:  # cap the heavy candidates per midnight: the least heavy become light
        h = sorted([c for c in by_k[k] if c.heavy], key=lambda c: -abs(np.dot(*c.pos_e[0]) - np.dot(*c.pos_e[1])))
        for c in h[MAX_HEAVY:]:
            c.heavy = False
    light = {k: [c for c in by_k[k] if not c.heavy] for k in by_k}
    heavy = {k: [c for c in by_k[k] if c.heavy] for k in by_k}
    pb = {k: _poisson_binomial(np.array([c.p_early for c in light[k]])) for k in by_k}
    states = {k: [(x, h) for x in range(len(light[k]) + 1) for h in itertools.product((0, 1), repeat=len(heavy[k]))] for k in by_k}  # h: 1 = earlier day
    prior = {k: np.array([pb[k][x] * np.prod([c.p_early if b else 1 - c.p_early for c, b in zip(heavy[k], h)]) for x, h in states[k]]) for k in by_k}

    # the delay-total density of each day, for the heavy candidates placed on it
    day_rest = {}
    for d in range(K):
        if not (heavy[d] or heavy.get(d + 1)) or days[d] not in pub_total or not np.isfinite(pub_total[days[d]]):
            continue
        rows = np.flatnonzero((base_day == d) & ~np.isin(np.arange(len(f)), [c.position for c in heavy[d] + heavy.get(d + 1, [])]))
        if not len(rows):
            continue
        sub = f.iloc[rows].reset_index(drop=True)
        p = prepare_day(sub["a"].to_numpy(dtype=float), sub["value"].to_numpy(dtype=float), resolve(sub), settings)
        share = float(noise_wide_share)
        noise = (1 - share) * _gauss_cf(noise_sd_s) + sum(0.5 * share * _gauss_cf(s) for s in settings.noise_wide_sd_s)
        cf = p.bulk_cf * np.prod(p.heavy_cf, axis=0) * noise * _gauss_cf(p.n_eff * tau_s) if len(p.heavy) else p.bulk_cf * noise * _gauss_cf(p.n_eff * tau_s)
        day_rest[d] = (cf, (pub_total[days[d]] - p.reference_total) / GRID_S)

    density_cache: dict = {}

    def density_ratio(d: int, h_d: tuple, h_n: tuple) -> float:
        key = (d, tuple(b for b in h_d), tuple(b for b in h_n))
        if key not in density_cache:
            extra = [c.pos_e[1] for c, b in zip(heavy[d], h_d) if not b] + [c.pos_e[0] for c, b in zip(heavy.get(d + 1, []), h_n) if b]
            cf, delta = day_rest[d]
            density_cache[key] = float(np.exp(_log_density(cf, extra, delta) - _log_density(cf, [], delta))) if extra else 1.0
        return density_cache[key]

    def day_factor(d: int, s_d: tuple, s_next: tuple) -> float:
        """Likelihood of day d given the state of midnight d (s_d: candidates between day d-1 and d) and of midnight d+1."""
        x_d, h_d = s_d
        x_n, h_n = s_next
        n = certain[d] + (len(light[d]) - x_d) + (len(h_d) - sum(h_d)) + x_n + sum(h_n)
        like = 1.0
        if days[d] in complete:
            off = n_pub[days[d]] - n
            like = (1 - eps) if off == 0 else (eps / 2 if abs(off) == 1 else 0.0)
        if like > 0 and d in day_rest:
            like *= density_ratio(d, h_d, h_n)
        return like

    # forward-backward over midnights 0..K (midnight K: after the last day, no candidates)
    factors = [np.array([[day_factor(d, s0, s1) for s1 in states[d + 1]] for s0 in states[d]]) for d in range(K)]
    fwd = [None] * (K + 1)
    fwd[0] = prior[0] / prior[0].sum()
    scale_log = 0.0
    for d in range(K):
        v = (fwd[d] @ factors[d]) * prior[d + 1]
        z = v.sum()
        if z <= 0:
            return value.copy(), {"status": "infeasible", "day": days[d], "candidates": len(cands), "skipped": skipped}
        fwd[d + 1] = v / z
        scale_log += np.log(z)
    bwd = [None] * (K + 1)
    bwd[K] = np.ones(len(states[K]))
    for d in range(K - 1, -1, -1):
        v = factors[d] @ (bwd[d + 1] * prior[d + 1])
        bwd[d] = v / v.sum()
    # a midnight's candidates move only where the published count of one of its two days disagrees with the values' days
    value_count = np.bincount(base_day[base_day >= 0], minlength=K)
    off_day = {d: bool(days[d] in complete and n_pub[days[d]] != value_count[d]) for d in range(K)}
    new = value.copy()
    moved = []
    for k in by_k:
        if not by_k[k] or not (off_day.get(k - 1, False) or off_day.get(k, False)):
            continue
        post = fwd[k] * bwd[k]
        post = post / post.sum()
        x_marg = np.zeros(len(light[k]) + 1)
        for (x, h), q in zip(states[k], post):
            x_marg[x] += q
        for i, c in enumerate(light[k]):
            rest = _poisson_binomial(np.array([o.p_early for j, o in enumerate(light[k]) if j != i]))
            full = pb[k]
            p_e = 0.0
            for x in range(1, len(full)):
                if full[x] > 0:
                    p_e += x_marg[x] * c.p_early * rest[x - 1] / full[x]
            moved.append((c, p_e))
        for i, c in enumerate(heavy[k]):
            moved.append((c, float(sum(q for (x, h), q in zip(states[k], post) if h[i] == 1))))
    every = [{"position": int(c.position), "k": int(c.k), "p_early_prior": float(c.p_early), "p_early_post": float(p_e), "heavy": bool(c.heavy), "big": bool(c.big),
              "value_early": bool(base_day[c.position] == c.k - 1), "flagged": True} for c, p_e in moved]
    kept = []
    for c, p_e in moved:
        value_early = base_day[c.position] == c.k - 1
        if c.big and (p_e > 0.5) == value_early:   # the counts leave a big candidate on its value's side of midnight: it keeps its value
            kept.append((c, p_e))
            continue
        new[c.position] = value[c.position] + (p_e - c.p_early) * (c.mean[0] - c.mean[1])
    moved = [(c, p_e) for c, p_e in moved if not any(c is k_ for k_, _ in kept)]
    new = np.where(np.isin(np.arange(len(f)), [c.position for c, _ in moved]), np.maximum(new, FLOOR_S), value)
    return new, {"status": "ok", "candidates": len(cands), "heavy": int(sum(c.heavy for c in cands)), "big": int(sum(c.big for c in cands)), "skipped": skipped,
                 "log_evidence": float(scale_log), "kept_big": len(kept), "flagged_candidates": every,
                 "moved": [{"position": int(c.position), "p_early_prior": float(c.p_early), "p_early_post": float(p_e), "heavy": bool(c.heavy), "big": bool(c.big),
                            "mean_early": float(c.mean[0]), "mean_late": float(c.mean[1])} for c, p_e in moved]}
