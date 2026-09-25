"""Thin a 2025 source month's adsb.lol ground points to a ranking month's coverage and re-run the FROZEN matcher (label-free).

For each declared airport the keep table (src/prc2026_regime/thinning.py) is estimated from ground points per departure in
the source and target months; each source day is thinned and its departures at those airports are matched again with
prc2026_adsb.features.match_departures under BOTH frozen stand tables (data/external/adsblol/matches, the committee's
1,332 stands, and data/external/adsblol/matches_v11, the residual stage's 1,111 stands). Rows of every other airport are
copied unchanged from the natural pass-2 tables. Output (git-ignored):
  data/external/adsblol/regime/<attempt>/{matches,matches_v11}/matches2_<source month>.parquet (+ stands.parquet copies)
  runs/coverage_regime/<attempt>/{keep_<airport>.parquet, attempt.json}
attempt.json records the regime statistics of natural source, thinned source and natural target per airport.
No label is read. Usage:
  regime_thin_match.py --attempt a1 --source 2025-12 --target 2026-01 --airports EDDF [--prior 50] [--verify-days 20251203]
"""
from __future__ import annotations

import argparse
import calendar
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TIME  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS, load_surface, match_departures  # noqa: E402
from prc2026_regime.thinning import ground_counts, keep_table, segment_counts, thin_surface  # noqa: E402

SURFACE = ROOT / "data/external/adsblol/surface"
DIRS = {"matches": ROOT / "data/external/adsblol/matches", "matches_v11": ROOT / "data/external/adsblol/matches_v11"}
REGIME = ROOT / "data/external/adsblol/regime"
RUNS = ROOT / "runs/coverage_regime"
LAT0 = {"EDDF": 50.03, "EDDM": 48.35, "EGLL": 51.47, "EHAM": 52.31, "LEBL": 41.30, "LEMD": 40.49, "LFPG": 49.01, "LIRF": 41.80, "LSZH": 47.46, "LTFM": 41.26}


def month_days(month: str) -> list[str]:
    y, m = int(month[:4]), int(month[5:7])
    return [f"{y}{m:02}{d:02}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]


def departures(month: str) -> pd.DataFrame:
    if month.startswith("2026"):
        raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=DEPARTURE_COLUMNS)
    else:
        y, m = int(month[:4]), int(month[5:7])
        raw = pd.read_parquet(ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet", columns=DEPARTURE_COLUMNS)
    return raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)


def counts(month: str, airports: list[str], density: str = "all") -> pd.DataFrame:
    if density == "departures":
        path = REGIME / "counts" / f"segments_v3_{month}.parquet"
        if path.exists():
            return pd.read_parquet(path).query("airport in @airports")
        matches = pd.read_parquet(DIRS["matches_v11"] / f"matches2_{month}.parquet")
        table = segment_counts(SURFACE, month_days(month), matches, list(LAT0), LAT0)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        return table[table["airport"].isin(airports)]
    path = REGIME / "counts" / f"ground_v3_{month}.parquet"
    if path.exists():
        table = pd.read_parquet(path)
        if set(airports) <= set(table["airport"].unique()):
            return table[table["airport"].isin(airports)]
    table = ground_counts(SURFACE, month_days(month), list(LAT0), LAT0)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table[table["airport"].isin(airports)]


def one_day(job):
    day, dep, keep, seed, stands, alpha, session_drop, rescale, session_gap = job
    surface = load_surface(SURFACE, [day])
    if surface.empty or dep.empty:
        return day, {k: pd.DataFrame() for k in stands}, {}
    thinned = thin_surface(surface, day, keep, LAT0, seed, alpha=alpha, session_drop=session_drop, session_rescale=rescale, session_gap_s=session_gap)
    kept = {a: [int((surface["airport"].eq(a) & surface["ground"]).sum()), int((thinned["airport"].eq(a) & thinned["ground"]).sum())] for a in keep["airport"].unique()}
    out = {}
    for name, table in stands.items():
        m = match_departures(dep, thinned, table)
        if len(m):
            m["day"] = day
        out[name] = m
    return day, out, kept


def natural_day(job):
    day, dep, stands = job
    surface = load_surface(SURFACE, [day])
    m = match_departures(dep, surface, stands)
    if len(m):
        m["day"] = day
    return m


def stats(dep: pd.DataFrame, matches: pd.DataFrame) -> dict:
    out = {}
    for airport, g in dep.groupby("ADEP_mvt"):
        x = matches[matches["airport"].eq(airport)] if len(matches) else matches
        n = len(g)
        dark = x[~x["gate"]] if len(x) else x
        out[airport] = {"departures": n, "match_pct": 100 * len(x) / n, "gate_pct": 100 * float(x["gate"].sum()) / n if len(x) else 0.0,
                        "points_median": float(x["points"].median()) if len(x) else None, "dark_points_median": float(dark["points"].median()) if len(dark) else None,
                        "dark_first_point_s_median": float((dark["takeoff"] - dark["segment_start"]).median()) if len(dark) else None,
                        "segment_s_median": float((x["last_ground"] - x["segment_start"]).median()) if len(x) else None,
                        "points_per_min_median": float(np.median(x["points"] / np.maximum(1.0, (x["last_ground"] - x["segment_start"]) / 60.0))) if len(x) else None}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--airports", nargs="+", required=True)
    parser.add_argument("--prior", type=float, default=50.0)
    parser.add_argument("--multiplier", default="", help="AIRPORT=m pairs scaling the calibrated ratios")
    parser.add_argument("--calibrate", action="store_true", help="scale the cell ratios so that the expected kept total equals the target total")
    parser.add_argument("--density", choices=["all", "departures"], default="all", help="estimate the ratios from all ground points or only from matched departure segments")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--session-drop", default="0", help="probability of removing an aircraft's whole ground session: one number, or AIRPORT=q pairs separated by commas")
    parser.add_argument("--no-session-rescale", action="store_true", help="do not raise cell probabilities to compensate the session drop")
    parser.add_argument("--session-gap", type=float, default=1800.0, help="gap (s) that splits an aircraft's ground points into sessions; 86400 = one session per aircraft-day")
    parser.add_argument("--alpha", default="0", help="clumping: share of the loss applied to whole 100 m cell visits (0 = independent points); one number or AIRPORT=a pairs")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--stand-tables", nargs="+", default=list(DIRS), choices=list(DIRS), help="which frozen stand tables to match with (default both)")
    parser.add_argument("--verify-days", nargs="*", default=[], help="days whose natural matching is re-run and compared with the stored natural tables")
    args = parser.parse_args()
    started = time.time()
    if "=" in args.session_drop:
        args.session_drop = {k: float(v) for k, v in (item.split("=") for item in args.session_drop.split(","))}
    else:
        args.session_drop = float(args.session_drop)
    def value(raw):
        return raw if raw == "decomposed" else float(raw)
    args.alpha = {k: value(v) for k, v in (item.split("=") for item in args.alpha.split(","))} if "=" in args.alpha else value(args.alpha)
    args.multiplier = {k: float(v) for k, v in (item.split("=") for item in args.multiplier.split(","))} if args.multiplier else {}
    out_runs = RUNS / args.attempt
    out_data = REGIME / args.attempt
    if (out_runs / f"attempt_{args.source}.json").exists():
        raise SystemExit(f"Preserve the existing attempt {args.attempt} for {args.source}")
    out_runs.mkdir(parents=True, exist_ok=True)
    dep_s, dep_t = departures(args.source), departures(args.target)
    mov_s = dep_s["ADEP_mvt"].value_counts().to_dict()
    mov_t = dep_t["ADEP_mvt"].value_counts().to_dict()
    keep, ratios = keep_table(counts(args.source, args.airports, args.density), counts(args.target, args.airports, args.density), mov_s, mov_t, prior=args.prior, calibrate=args.calibrate, multiplier=args.multiplier)
    for airport, table in keep.groupby("airport"):
        table.to_parquet(out_runs / f"keep_{args.source}_{airport}.parquet", index=False)
    stands = {name: pd.read_parquet(d / "stands.parquet") for name, d in DIRS.items() if name in args.stand_tables}
    dep_s["day"] = dep_s[TIME].dt.strftime("%Y%m%d")
    target_dep = dep_s[dep_s["ADEP_mvt"].isin(args.airports)]
    jobs = [(day, target_dep[target_dep["day"].eq(day)].drop(columns=["day"]).reset_index(drop=True), keep, args.seed, stands, args.alpha, args.session_drop, not args.no_session_rescale, args.session_gap) for day in month_days(args.source)]
    with mp.Pool(min(args.workers, 12)) as pool:
        results = pool.map(one_day, jobs)
    kept = {}
    thinned = {name: [] for name in stands}
    for day, frames, k in results:
        kept[day] = k
        for name, frame in frames.items():
            if len(frame):
                thinned[name].append(frame)
    record = {"created_at_utc": utc_now(), "attempt": args.attempt, "source": args.source, "target": args.target, "airports": args.airports, "prior": args.prior, "density": args.density, "calibrate": args.calibrate, "multiplier": args.multiplier,
              "seed": args.seed, "alpha": args.alpha, "session_drop": args.session_drop, "session_rescale": not args.no_session_rescale, "session_gap_s": args.session_gap, "airport_point_ratio_target_over_source_per_departure": ratios, "labels_read": False,
              "keep_probability_summary": {a: {"cells": int(len(t)), "mean_keep_weighted_by_source_points": float(np.average(t["keep"], weights=t["n_s"])),
                                               "share_cells_keep_below_0.2": float((t["keep"] < 0.2).mean())} for a, t in keep.groupby("airport")},
              "ground_points_kept": {a: [sum(kept[d].get(a, [0, 0])[0] for d in kept), sum(kept[d].get(a, [0, 0])[1] for d in kept)] for a in args.airports},
              "outputs": {}, "stats": {}}
    for name, directory in DIRS.items():
        if name not in stands:
            continue
        natural = pd.read_parquet(directory / f"matches2_{args.source}.parquet")
        new = pd.concat(thinned[name], ignore_index=True) if thinned[name] else natural.iloc[:0]
        merged = pd.concat([natural[~natural["airport"].isin(args.airports)], new], ignore_index=True)
        target_dir = out_data / name
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"matches2_{args.source}.parquet"
        merged.to_parquet(path, index=False, compression="zstd")
        shutil.copyfile(directory / "stands.parquet", target_dir / "stands.parquet")
        record["outputs"][name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "rows": int(len(merged)),
                                   "natural_sha256": sha256_file(directory / f"matches2_{args.source}.parquet"), "stands_sha256": sha256_file(directory / "stands.parquet")}
        if name == "matches_v11":
            natural_target = pd.read_parquet(directory / f"matches2_{args.target}.parquet")
            record["stats"] = {"natural_source": stats(dep_s, natural), "thinned_source": stats(dep_s, merged), "natural_target": stats(dep_t, natural_target)}
    if args.verify_days:
        checks = {}
        for day in args.verify_days:
            dep = target_dep[target_dep["day"].eq(day)].drop(columns=["day"]).reset_index(drop=True)
            for name, directory in [(n, d) for n, d in DIRS.items() if n in stands]:
                rebuilt = natural_day((day, dep, stands[name])).sort_values(ID).reset_index(drop=True)
                stored = pd.read_parquet(directory / f"matches2_{args.source}.parquet")
                stored = stored[stored["airport"].isin(args.airports) & stored["day"].astype(str).str.replace("-", "").eq(day)].sort_values(ID).reset_index(drop=True)
                cols = [c for c in stored.columns if c in rebuilt.columns]
                same = len(rebuilt) == len(stored) and all(np.allclose(rebuilt[c].to_numpy(dtype=float), stored[c].to_numpy(dtype=float), equal_nan=True, rtol=0, atol=1e-9)
                                                          if pd.api.types.is_numeric_dtype(stored[c]) else rebuilt[c].astype(str).equals(stored[c].astype(str)) for c in cols)
                checks[f"{day}/{name}"] = {"rows_rebuilt": int(len(rebuilt)), "rows_stored": int(len(stored)), "identical": bool(same)}
        record["natural_reproduction_check"] = checks
    record["seconds"] = round(time.time() - started, 1)
    record["code_sha256"] = {p: sha256_file(ROOT / p) for p in ["tools/regime_thin_match.py", "src/prc2026_regime/thinning.py", "src/prc2026_adsb/features.py"]}
    write_json(out_runs / f"attempt_{args.source}.json", record)
    s = record["stats"]
    for a in args.airports:
        print(a, {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in s[k][a].items() if kk != "departures"} for k in ["natural_source", "thinned_source", "natural_target"]})
    print({"ratios": ratios, "kept": record["ground_points_kept"], "verify": record.get("natural_reproduction_check"), "seconds": record["seconds"]})


if __name__ == "__main__":
    main()
