"""Spatial thinning of adsb.lol ground points from a source month's coverage to a target month's coverage (label-free).

The ranking months lost receivers near some aprons (EDDF, LSZH in July, LIRF in July): ground points per movement fell
unevenly over the airport. For each airport the transform estimates, on a 100 m grid, the ratio of target to source ground
points per movement (departures + arrivals of that airport and month), shrinks each cell's ratio toward its 400 m parent
cell and that toward the airport-wide ratio (pseudo-count `prior` source points at each level), and keeps each source
ground point with probability min(1, ratio) of its cell. Airborne points are never removed. The random draw is seeded by
(seed, day, airport), so a thinned day is reproducible.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG = 111320.0


def cell_keys(lat: np.ndarray, lon: np.ndarray, lat0: float, size_m: float) -> tuple[np.ndarray, np.ndarray]:
    dy = size_m / M_PER_DEG
    dx = size_m / (M_PER_DEG * np.cos(np.radians(lat0)))
    return np.floor(lat / dy).astype(np.int64), np.floor(lon / dx).astype(np.int64)


def _cell_counts(g: pd.DataFrame, lat0: float, sizes, visit_gap_s: float) -> list[pd.DataFrame]:
    """Points (n) and visits (v) per cell for each size; a visit is an aircraft's run of consecutive points in one cell of that
    size (a new run starts when the cell changes or after more than visit_gap_s)."""
    g = g.sort_values(["icao", "time"], kind="mergesort")
    icao, tt = g["icao"].to_numpy(), g["time"].to_numpy()
    out = []
    for size in sizes:
        cy, cx = cell_keys(g["lat"].to_numpy(), g["lon"].to_numpy(), lat0, size)
        new = np.ones(len(g), dtype=bool)
        new[1:] = (icao[1:] != icao[:-1]) | (cy[1:] != cy[:-1]) | (cx[1:] != cx[:-1]) | (np.diff(tt) > visit_gap_s)
        c = pd.DataFrame({"cy": cy, "cx": cx, "n": 1, "v": new.astype(np.int64)}).groupby(["cy", "cx"], as_index=False)[["n", "v"]].sum()
        c["size"] = size
        out.append(c)
    return out


def ground_counts(surface_dir: Path, days: list[str], airports: list[str], lat0: dict[str, float], sizes=(100.0, 400.0),
                  visit_gap_s: float = 60.0) -> pd.DataFrame:
    """Ground points (n) and 100 m cell visits (v) per (airport, size, cell) over the given days.

    Visits are counted per size (see _cell_counts): a 400 m cell's visits are runs in that 400 m cell, not sums of 100 m visits.
    """
    parts = []
    for day in days:
        path = surface_dir / f"surface_{day}.parquet"
        if not path.exists():
            continue
        s = pd.read_parquet(path, columns=["icao", "time", "airport", "lat", "lon", "ground"])
        s = s[s["ground"] & s["airport"].isin(airports)]
        for airport, g in s.groupby("airport"):
            for c in _cell_counts(g, lat0[airport], sizes, visit_gap_s):
                c["airport"] = airport
                parts.append(c)
    out = pd.concat(parts, ignore_index=True)
    return out.groupby(["airport", "size", "cy", "cx"], as_index=False)[["n", "v"]].sum()


def segment_counts(surface_dir: Path, days: list[str], matches: pd.DataFrame, airports: list[str], lat0: dict[str, float],
                   sizes=(100.0, 400.0), visit_gap_s: float = 60.0) -> pd.DataFrame:
    """Like ground_counts, but only the ground points inside matched departure segments (the matched aircraft's points from
    segment_start to last_ground): the coverage of departure taxi routes, which is what the departure features see."""
    parts = []
    for day in days:
        path = surface_dir / f"surface_{day}.parquet"
        if not path.exists():
            continue
        m = matches[matches["airport"].isin(airports) & matches["day"].astype(str).str.replace("-", "").eq(day)]
        if m.empty:
            continue
        s = pd.read_parquet(path, columns=["icao", "time", "airport", "lat", "lon", "ground"])
        s = s[s["ground"] & s["airport"].isin(airports)]
        for airport, g in s.groupby("airport"):
            seg = m[m["airport"].eq(airport)][["icao", "segment_start", "last_ground"]]
            j = g.merge(seg, on="icao", how="inner")
            j = j[(j["time"] >= j["segment_start"]) & (j["time"] <= j["last_ground"])].drop_duplicates(["icao", "time", "lat", "lon"])
            if j.empty:
                continue
            for c in _cell_counts(j, lat0[airport], sizes, visit_gap_s):
                c["airport"] = airport
                parts.append(c)
    out = pd.concat(parts, ignore_index=True)
    return out.groupby(["airport", "size", "cy", "cx"], as_index=False)[["n", "v"]].sum()


def _shrunk_ratio(s: pd.DataFrame, t: pd.DataFrame, col: str, scale: float, prior: float) -> tuple[pd.DataFrame, float]:
    r_air = float(t[col].sum() * scale / s[col].sum())
    coarse = s[s["size"].eq(400.0)].merge(t[t["size"].eq(400.0)], on=["cy", "cx"], how="left", suffixes=("_s", "_t")).fillna({f"{col}_t": 0.0})
    coarse["r400"] = (coarse[f"{col}_t"] * scale + prior * r_air) / (coarse[f"{col}_s"] + prior)
    fine = s[s["size"].eq(100.0)].merge(t[t["size"].eq(100.0)], on=["cy", "cx"], how="left", suffixes=("_s", "_t")).fillna({f"{col}_t": 0.0})
    fine["py"], fine["px"] = np.floor(fine["cy"] / 4).astype(np.int64), np.floor(fine["cx"] / 4).astype(np.int64)
    fine = fine.merge(coarse[["cy", "cx", "r400"]].rename(columns={"cy": "py", "cx": "px"}), on=["py", "px"], how="left")
    fine["r400"] = fine["r400"].fillna(r_air)
    fine["ratio"] = (fine[f"{col}_t"] * scale + prior * fine["r400"]) / (fine[f"{col}_s"] + prior)
    return fine[["cy", "cx", f"{col}_s", f"{col}_t", "ratio"]], r_air


def _calibrate(weights: np.ndarray, ratio: np.ndarray, total: float) -> float:
    """c such that sum(weights * min(1, c * ratio)) equals `total`. If the target total is not below the source total (a source
    month with missing archive days), every point is kept: a large finite c (1e9; infinity would turn a zero ratio into NaN)."""
    if total >= weights.sum():
        return 1e9
    lo, hi = 0.0, 1.0
    while (weights * np.minimum(1.0, hi * ratio)).sum() < total:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (weights * np.minimum(1.0, mid * ratio)).sum() < total:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def keep_table(source: pd.DataFrame, target: pd.DataFrame, movements_source: dict[str, int], movements_target: dict[str, int],
               prior: float = 50.0, calibrate: bool = False, multiplier: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Keep probabilities per (airport, 100 m cell) and the airport-wide ratios.

    keep = min(1, point ratio) (target over source ground points per departure, 100 m cells, shrunk toward the 400 m parent
    and that toward the airport ratio with `prior` source points of pseudo-count). The decomposition splits it into coverage
    and rate: keep_visit = min(1, visit ratio of the 400 m parent cell) -- a taxiing aircraft needs about 50 s to cross 400 m,
    so a lower point rate does not hide a visit there as it does in a 100 m cell -- and keep_point = min(1, point ratio /
    keep_visit). Where the target covers some cells better than the source (ratio > 1, which thinning cannot reproduce),
    `calibrate` multiplies the point ratios (and separately the visit ratios) by the one factor per airport that makes the
    expected kept total equal the target total, instead of losing the clipped mass. `multiplier` (per airport) scales both
    factors further, to match the target's class shares (matched / gate) where coverage moved rather than shrank.
    """
    tables, ratios = [], {}
    for airport in sorted(source["airport"].unique()):
        scale = movements_source[airport] / movements_target[airport]  # target counts expressed at the source month's traffic
        s = source[source["airport"].eq(airport)]
        t = target[target["airport"].eq(airport)]
        points, r_points = _shrunk_ratio(s, t, "n", scale, prior)
        s400, t400 = s[s["size"].eq(400.0)], t[t["size"].eq(400.0)]
        r_visits = float(t400["v"].sum() * scale / s400["v"].sum())
        coarse = s400.merge(t400, on=["cy", "cx"], how="left", suffixes=("_s", "_t")).fillna({"v_t": 0.0})
        coarse["ratio_visits"] = (coarse["v_t"] * scale + prior * r_visits) / (coarse["v_s"] + prior)
        fine = points.rename(columns={"ratio": "ratio_points"})
        fine["py"], fine["px"] = np.floor(fine["cy"] / 4).astype(np.int64), np.floor(fine["cx"] / 4).astype(np.int64)
        fine = fine.merge(coarse[["cy", "cx", "ratio_visits"]].rename(columns={"cy": "py", "cx": "px"}), on=["py", "px"], how="left")
        fine["ratio_visits"] = fine["ratio_visits"].fillna(r_visits)
        c_points = c_visits = 1.0
        if calibrate:
            c_points = _calibrate(fine["n_s"].to_numpy(float), fine["ratio_points"].to_numpy(float), float(fine["n_t"].sum() * scale))
            c_visits = _calibrate(coarse["v_s"].to_numpy(float), coarse["ratio_visits"].to_numpy(float), float(coarse["v_t"].sum() * scale))
        extra = float((multiplier or {}).get(airport, 1.0))
        c_points, c_visits = c_points * extra, c_visits * extra
        fine["keep"] = (c_points * fine["ratio_points"]).clip(0.0, 1.0)
        fine["keep_visit"] = (c_visits * fine["ratio_visits"]).clip(0.0, 1.0)
        fine["keep_point"] = (fine["ratio_points"] / fine["keep_visit"].clip(lower=1e-6)).clip(0.0, 1.0)
        fine["airport"] = airport
        ratios[airport] = {"points": r_points, "visits": r_visits, "calibration_points": c_points, "calibration_visits": c_visits}
        tables.append(fine)
    return pd.concat(tables, ignore_index=True), ratios


def _seed(seed: int, day: str, airport: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{day}|{airport}".encode()).digest()[:8], "little")


def thin_surface(surface: pd.DataFrame, day: str, keep: pd.DataFrame, lat0: dict[str, float], seed: int, alpha: float | dict | str = 0.0,
                 visit_gap_s: float = 60.0, session_drop: float | dict = 0.0, session_gap_s: float = 1800.0,
                 session_rescale: bool = True) -> pd.DataFrame:
    """The day's surface points with ground points of the keep table's airports thinned; every other row is unchanged.

    alpha = 0 keeps each ground point independently with its cell's probability p. alpha > 0 makes the loss clumped, as when
    an aircraft is either in a receiver's line of sight or not: an aircraft's consecutive points in one 100 m cell (a cell
    visit; a new visit starts when the cell changes or after a gap of more than visit_gap_s) are kept together with
    probability p ** alpha, and the points of a kept visit independently with p ** (1 - alpha). The expected share kept is p
    either way. alpha = "decomposed" keeps an aircraft's run of points in one 400 m cell with that cell's keep_visit and each
    point of a kept run with its 100 m cell's keep_point (both estimated from the two months, see keep_table). session_drop = q > 0 additionally removes an aircraft's whole ground session (its ground points at the
    airport, split where consecutive points are more than session_gap_s apart) with probability q -- the ranking months have
    departures never seen on the ground at all -- and, with session_rescale, raises every cell's probability to
    min(1, p / (1 - q)) to keep the expected density (without it, kept sessions are thinned exactly as with q = 0).
    """
    if surface.empty:
        return surface
    drop = np.zeros(len(surface), dtype=bool)
    for airport, table in keep.groupby("airport"):
        rows = np.flatnonzero(surface["airport"].eq(airport).to_numpy() & surface["ground"].to_numpy())
        if len(rows) == 0:
            continue
        q = float(session_drop.get(airport, 0.0)) if isinstance(session_drop, dict) else float(session_drop)
        a = alpha.get(airport, 0.0) if isinstance(alpha, dict) else alpha
        decomposed = a == "decomposed"
        a = 1.0 if decomposed else float(a)
        sub = surface.iloc[rows]
        cy, cx = cell_keys(sub["lat"].to_numpy(), sub["lon"].to_numpy(), lat0[airport], 100.0)
        lookup = table.set_index(["cy", "cx"])["keep"]
        index = pd.MultiIndex.from_arrays([cy, cx])
        p = lookup.reindex(index).to_numpy(dtype=float)
        p = np.where(np.isfinite(p), p, float(table["keep"].median()))
        if decomposed:
            pv = table.set_index(["cy", "cx"])["keep_visit"].reindex(index).to_numpy(dtype=float)
            pp = table.set_index(["cy", "cx"])["keep_point"].reindex(index).to_numpy(dtype=float)
            pv = np.where(np.isfinite(pv), pv, float(table["keep_visit"].median()))
            pp = np.where(np.isfinite(pp), pp, float(table["keep_point"].median()))
        rng = np.random.default_rng(_seed(seed, day, airport))
        if a <= 0 and q <= 0:
            drop[rows] = rng.random(len(rows)) >= p
            continue
        order = np.lexsort((sub["time"].to_numpy(), sub["icao"].to_numpy()))
        icao, t = sub["icao"].to_numpy()[order], sub["time"].to_numpy()[order]
        cyo, cxo, po = cy[order], cx[order], p[order]
        visit_p, point_p = (pv[order], pp[order]) if decomposed else (None, None)
        kept = np.ones(len(order), dtype=bool)
        if q > 0:
            if session_rescale:
                po = np.minimum(1.0, po / (1.0 - q))
            start = np.ones(len(order), dtype=bool)
            start[1:] = (icao[1:] != icao[:-1]) | (np.diff(t) > session_gap_s)
            session = np.cumsum(start) - 1
            kept &= rng.random(int(session[-1]) + 1)[session] >= q
        if decomposed:  # visits are runs in the 400 m parent cell, the level keep_visit was estimated at
            cyo, cxo = np.floor(cyo / 4).astype(np.int64), np.floor(cxo / 4).astype(np.int64)
        new = np.ones(len(order), dtype=bool)
        new[1:] = (icao[1:] != icao[:-1]) | (cyo[1:] != cyo[:-1]) | (cxo[1:] != cxo[:-1]) | (np.diff(t) > visit_gap_s)
        visit = np.cumsum(new) - 1
        visit_u = rng.random(int(visit[-1]) + 1)
        point_u = rng.random(len(order))
        if decomposed:
            kept &= (visit_u[visit] < visit_p) & (point_u < point_p)
        else:
            kept &= (visit_u[visit] < po ** a) & (point_u < po ** (1.0 - a))
        drop[rows[order]] = ~kept
    return surface.loc[~drop].reset_index(drop=True)
