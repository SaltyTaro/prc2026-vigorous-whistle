"""Label-free feature tables of a source month in a thinned coverage regime (or, with --natural, the natural month as a check).

From one thinning attempt of tools/regime_thin_match.py, for one source month:
  committee  run_adsb_learner.adsb_table on the attempt's thinned `matches` directory (departure + neighbour features)
  residual   run_adsb_v24.adsb_features on the attempt's thinned `matches_v11` directory
  v19        the eleven v19 features (tools/build_adsb_v19_features.py's steps, re-run here with every day's surface thinned
             exactly as in the attempt: same keep tables, parameters and seeds), on the thinned `matches` tables
Output: runs/coverage_regime/<attempt>/features/{committee,residual,v19}/... No label is read. With --natural the same code
runs on the natural surface and natural tables and must reproduce the stored v19 table (the check is recorded).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TIME  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS, EPOCH, load_surface  # noqa: E402
from prc2026_adsb.features_v19 import (CONGESTION_NAMES, FEATURE_NAMES_V19, TIMING_NAMES, impute_stands, stand_features,  # noqa: E402
                                        surface_congestion, timing_features)
from prc2026_regime.thinning import thin_surface  # noqa: E402
from build_adsb_v19_features import raw_path  # noqa: E402
from regime_thin_match import LAT0, REGIME, RUNS, SURFACE  # noqa: E402

NATURAL = {"matches": ROOT / "data/external/adsblol/matches", "matches_v11": ROOT / "data/external/adsblol/matches_v11"}


def thinning_spec(attempt: str, month: str) -> dict | None:
    record = read_json(RUNS / attempt / f"attempt_{month}.json")
    keep = pd.concat([pd.read_parquet(RUNS / attempt / f"keep_{month}_{a}.parquet").assign(airport=a) for a in record["airports"]], ignore_index=True)
    return {"keep": keep, "seed": record["seed"], "alpha": record.get("alpha", 0.0), "session_drop": record.get("session_drop", 0.0) or 0.0,
            "session_rescale": record.get("session_rescale", True), "session_gap_s": record.get("session_gap_s", 1800.0), "record": record}


def surface_for(day: str, spec: dict | None) -> pd.DataFrame:
    surface = load_surface(SURFACE, [day])
    if spec is None or surface.empty:
        return surface
    return thin_surface(surface, day, spec["keep"], LAT0, spec["seed"], alpha=spec["alpha"], session_drop=spec["session_drop"],
                        session_rescale=spec["session_rescale"], session_gap_s=spec["session_gap_s"])


def pass_one(args):
    day, matches, spec = args
    surface = surface_for(day, spec)
    if surface.empty or matches.empty:
        return pd.DataFrame(columns=[ID, *TIMING_NAMES]), pd.DataFrame(columns=["airport", "runway", "lat", "lon"])
    timing = timing_features(matches, surface)
    by_icao = {i: frame for i, frame in surface.groupby("icao")}
    rows = []
    for r in matches.itertuples(index=False):
        track = by_icao.get(r.icao)
        if track is None:
            continue
        roll = track[track["ground"] & (track["time"] >= r.last_ground - 180) & (track["time"] <= r.last_ground) & (track["gs"].fillna(0) >= 30)].sort_values("time")
        if len(roll):
            rows.append((r.airport, r.RUNWAY_mvt, float(roll["lat"].iloc[0]), float(roll["lon"].iloc[0])))
    return timing, pd.DataFrame(rows, columns=["airport", "runway", "lat", "lon"])


def pass_two(args):
    day, departures, liftoff, own, spec = args
    surface = surface_for(day, spec)
    covered = set(surface["airport"].unique()) if len(surface) else set()
    dep = departures[departures["ADEP_mvt"].isin(covered)]
    if dep.empty:
        return pd.DataFrame(columns=[ID, *CONGESTION_NAMES])
    return surface_congestion(dep, surface, liftoff, own_icao=own)


def v19_table(month: str, matches_dir: Path, spec: dict | None, workers: int) -> pd.DataFrame:
    """tools/build_adsb_v19_features.py's main loop for one month, with the surface thinned by `spec` (None = natural)."""
    stands = pd.read_parquet(matches_dir / "stands.parquet")
    raw = pd.read_parquet(raw_path(month), columns=DEPARTURE_COLUMNS + ["RUNWAY_mvt"])
    dep = raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
    dep["takeoff"] = (dep[TIME] - EPOCH).dt.total_seconds()
    dep["day"] = dep[TIME].dt.strftime("%Y%m%d")
    matches = pd.read_parquet(matches_dir / f"matches2_{month}.parquet").merge(dep[[ID, "RUNWAY_mvt"]], on=ID, how="inner")
    matches["day"] = matches["day"].astype(str).str.replace("-", "")
    days = sorted(dep["day"].unique())
    with mp.Pool(workers) as pool:
        first = pool.map(pass_one, [(day, matches[matches["day"].eq(day)].reset_index(drop=True), spec) for day in days])
    timing = pd.concat([t for t, _ in first], ignore_index=True)
    points = pd.concat([p for _, p in first], ignore_index=True)
    liftoff = points.groupby(["airport", "runway"]).agg(lat=("lat", "median"), lon=("lon", "median"), n=("lat", "size")).reset_index()
    liftoff = liftoff[liftoff["n"] >= 20].reset_index(drop=True)
    own = dep[[ID]].merge(matches[[ID, "icao"]], on=ID, how="left")["icao"]
    own.index = dep.index
    with mp.Pool(workers) as pool:
        second = pool.map(pass_two, [(day, dep[dep["day"].eq(day)], liftoff, own[dep["day"].eq(day)], spec) for day in days])
    congestion = pd.concat(second, ignore_index=True)
    positions = impute_stands(stands, dep)
    stand = stand_features(matches, positions)
    out = dep[[ID]].merge(timing, on=ID, how="left").merge(stand, on=ID, how="left").merge(congestion, on=ID, how="left")
    return out[[ID, *FEATURE_NAMES_V19]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--natural", action="store_true", help="build from the natural surface and tables (reproduction check)")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    from run_adsb_learner import adsb_table, source_file
    from run_adsb_v24 import adsb_features
    started = time.time()
    workers = min(args.workers, 12)
    label = "natural" if args.natural else "thinned"
    out = RUNS / args.attempt / "features" / label
    spec = None if args.natural else thinning_spec(args.attempt, args.month)
    dirs = NATURAL if args.natural else {name: REGIME / args.attempt / name for name in NATURAL}
    record = {"created_at_utc": utc_now(), "attempt": args.attempt, "month": args.month, "kind": label, "labels_read": False, "outputs": {}}
    stands = pd.read_parquet(dirs["matches"] / "stands.parquet")
    committee = adsb_table([args.month], source_file(args.month), dirs["matches"], stands, out / "committee")
    record["outputs"]["committee"] = sha256_file(out / "committee" / f"adsb_{source_file(args.month).replace('.parquet', '')}.parquet")
    residual = adsb_features(args.month, dirs["matches_v11"], out / "residual")
    record["outputs"]["residual"] = sha256_file(out / "residual" / f"matches_v11_{args.month}.parquet")
    v19 = v19_table(args.month, dirs["matches"], spec, workers)
    (out / "v19").mkdir(parents=True, exist_ok=True)
    v19.to_parquet(out / "v19" / f"features_v19_{args.month}.parquet", index=False)
    record["outputs"]["v19"] = sha256_file(out / "v19" / f"features_v19_{args.month}.parquet")
    record["rows"] = {"committee": int(len(committee)), "residual": int(len(residual)), "v19": int(len(v19))}
    record["matched_share"] = {"committee": float(committee["adsb_matched"].mean()), "residual": float(residual["adsb_matched"].mean())}
    if args.natural:
        stored = pd.read_parquet(ROOT / "data/external/adsblol/features_v19" / f"features_v19_{args.month}.parquet").set_index(ID).reindex(v19[ID])
        rebuilt = v19.set_index(ID)
        record["v19_reproduction_max_abs_diff"] = {c: float(np.nanmax(np.abs(rebuilt[c].to_numpy(float) - stored[c].to_numpy(float)))) if np.isfinite(rebuilt[c].to_numpy(float)).any() else 0.0 for c in FEATURE_NAMES_V19}
        record["v19_reproduction_nan_pattern_equal"] = {c: bool(np.array_equal(np.isnan(rebuilt[c].to_numpy(float)), np.isnan(stored[c].to_numpy(float)))) for c in FEATURE_NAMES_V19}
        for name, cached in [("committee", ROOT / "runs/adsb_nn/features" / f"adsb_{source_file(args.month).replace('.parquet', '')}.parquet"),
                             ("residual", ROOT / "runs/adsb_v24/features" / f"matches_v11_{args.month}.parquet")]:
            if cached.exists():
                a = (committee if name == "committee" else residual).set_index(ID)
                b = pd.read_parquet(cached).set_index(ID).reindex(a.index)
                cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
                text = [c for c in a.columns if c in b.columns and c not in cols]
                record[f"{name}_reproduction_max_abs_diff"] = float(max(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))) if np.isfinite(a[c].to_numpy(float)).any() else 0.0 for c in cols))
                record[f"{name}_reproduction_nan_pattern_equal"] = bool(all(np.array_equal(np.isnan(a[c].to_numpy(float)), np.isnan(b[c].to_numpy(float))) for c in cols))
                record[f"{name}_reproduction_text_columns_equal"] = {c: bool(a[c].astype(str).equals(b[c].astype(str))) for c in text}
                record[f"{name}_reference"] = cached.relative_to(ROOT).as_posix()
    record["seconds"] = round(time.time() - started, 1)
    record["code_sha256"] = {p: sha256_file(ROOT / p) for p in ["tools/regime_features.py", "src/prc2026_regime/thinning.py", "src/prc2026_adsb/features.py", "src/prc2026_adsb/features_v19.py"]}
    write_json(out / f"features_{args.month}.json", record)
    print({k: record[k] for k in record if k not in ("outputs", "code_sha256")})


if __name__ == "__main__":
    main()
