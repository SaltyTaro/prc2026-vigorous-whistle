"""Scratch screen (lever B): richer features from the raw adsb.lol ground trace of each matched departure, development months only.
Writes runs/clean_line_screens/trace/features_<month>.parquet. Label-free; nothing exported."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/clean_line_screens/trace"
ID = "MVT_ID_mvt"
MONTHS = sys.argv[1:] or ["2025-01", "2025-07", "2025-11", "2025-12"]
MARKS = [60, 120, 300, 600, 900, 1200, 1800]


def features(t, x, y, gs):
    """t seconds relative to takeoff (ascending, <= 0), x/y metres, gs knots (may be NaN)."""
    n = len(t)
    out = {"tr_points": n, "tr_first_s": -t[0], "tr_last_s": -t[-1]}
    if n < 3:
        return out
    dt = np.diff(t)
    step = np.hypot(np.diff(x), np.diff(y))
    ok = dt > 0
    v = np.where(ok, step / np.where(ok, dt, 1.0), 0.0)
    out["tr_gap_max_s"] = float(dt.max())
    out["tr_gap_mean_s"] = float(dt.mean())
    out["tr_path_m"] = float(step.sum())
    straight = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    out["tr_straight_m"] = straight
    out["tr_ratio"] = out["tr_path_m"] / max(straight, 1.0)
    still = v < 0.5
    out["tr_still_s"] = float(dt[still].sum())
    out["tr_move_speed"] = float(step[~still].sum() / max(dt[~still].sum(), 1.0))
    # runs of standing still
    runs, start = [], None
    for i, s in enumerate(still):
        if s and start is None:
            start = i
        if (not s) and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(still)))
    durations = [t[b] - t[a] for a, b in runs]
    out["tr_stops30"] = int(sum(d >= 30 for d in durations))
    out["tr_stops120"] = int(sum(d >= 120 for d in durations))
    if durations:
        k = int(np.argmax(durations))
        a, b = runs[k]
        out["tr_longest_stop_s"] = float(durations[k])
        out["tr_longest_stop_end_s"] = float(-t[b])
        out["tr_longest_stop_dist_m"] = float(np.hypot(x[a] - x[-1], y[a] - y[-1]))
        a, b = runs[-1]
        out["tr_last_stop_s"] = float(durations[-1])
        out["tr_last_stop_end_s"] = float(-t[b])
        a, b = runs[0]
        out["tr_first_stop_s"] = float(durations[0])
        out["tr_first_stop_start_s"] = float(-t[a])
    mid = 0.5 * (t[1:] + t[:-1])
    for w in (300, 600, 900):
        k = mid >= -w
        out[f"tr_still_last{w}_s"] = float(dt[k & still].sum())
        out[f"tr_path_last{w}_m"] = float(step[k].sum())
    dist_end = np.hypot(x - x[-1], y - y[-1])
    for m in MARKS:
        out[f"tr_dist_at_{m}_m"] = float(np.interp(-m, t, dist_end)) if -m >= t[0] else np.nan
    g = gs[np.isfinite(gs)]
    if len(g):
        out["tr_gs_p10"], out["tr_gs_p50"], out["tr_gs_p90"], out["tr_gs_max"] = (float(q) for q in (*np.quantile(g, [0.1, 0.5, 0.9]), g.max()))
    third = max(1, len(v) // 3)
    out["tr_speed_first_third"] = float(step[:third].sum() / max(dt[:third].sum(), 1.0))
    out["tr_speed_last_third"] = float(step[-third:].sum() / max(dt[-third:].sum(), 1.0))
    return out


for month in MONTHS:
    started = time.time()
    matches = pd.read_parquet(ROOT / f"data/external/adsblol/matches/matches2_{month}.parquet", columns=[ID, "airport", "takeoff", "segment_start", "icao", "day", "anchor_lat", "anchor_lon"])
    rows = []
    for day, md in matches.groupby("day"):
        path = ROOT / f"data/external/adsblol/surface/surface_{day}.parquet"
        if not path.exists():
            continue
        pts = pd.read_parquet(path, columns=["icao", "time", "lat", "lon", "ground", "gs"])
        pts = pts[pts["ground"] & pts["icao"].isin(md["icao"])]
        j = pts.merge(md, on="icao", how="inner")
        j = j[(j["time"] >= j["segment_start"] - 1.0) & (j["time"] <= j["takeoff"])].sort_values([ID, "time"], kind="stable")
        lat0 = np.deg2rad(j["anchor_lat"].to_numpy())
        j["x"] = (j["lon"].to_numpy() - j["anchor_lon"].to_numpy()) * 111320.0 * np.cos(lat0)
        j["y"] = (j["lat"].to_numpy() - j["anchor_lat"].to_numpy()) * 110540.0
        j["t"] = j["time"].to_numpy() - j["takeoff"].to_numpy()
        ids = j[ID].to_numpy()
        bounds = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
        t, x, y, gs = (j[c].to_numpy(dtype=float) for c in ("t", "x", "y", "gs"))
        for a, b in zip(bounds[:-1], bounds[1:]):
            f = features(t[a:b], x[a:b], y[a:b], gs[a:b])
            f[ID] = ids[a]
            rows.append(f)
    table = pd.DataFrame(rows)
    table.to_parquet(OUT / f"features_{month}.parquet", index=False)
    print(month, {"matches": len(matches), "with_trace": len(table), "columns": table.shape[1], "seconds": round(time.time() - started, 1)}, flush=True)
