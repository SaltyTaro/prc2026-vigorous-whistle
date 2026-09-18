"""Build the v19 screen features (docs/adsb_v19_screen/README.md) for whole months, day-parallel.

Pass 1 (per day): segment-timing features for matched departures and the take-off roll
start (first point at 30 kt or more in the last 3 minutes on the ground) of each matched
departure, whose per airport-runway median is the runway queue point. Pass 2 (per day): surface
congestion for every departure at an airport with surface data that day. Stand
imputation is month-level. Output: data/external/adsblol/features_v19/features_v19_<month>.parquet
with MVT_ID_mvt and the eleven features (NaN where not applicable). Label-free.

    python tools/build_adsb_v19_features.py --months 2025-01 2025-07 2025-11 --workers 6
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from prc2026.features import ID, PHASE, TIME  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS, EPOCH, load_surface  # noqa: E402
from prc2026_adsb.features_v19 import (CONGESTION_NAMES, FEATURE_NAMES_V19, TIMING_NAMES, impute_stands, stand_features,  # noqa: E402
                                        surface_congestion, timing_features)
from build_adsb_features import raw_path  # noqa: E402

SURFACE = ROOT / "data/external/adsblol/surface"
MATCHES = ROOT / "data/external/adsblol/matches_v11"
OUTPUT = ROOT / "data/external/adsblol/features_v19"


def pass_one(args):
    day, matches = args
    surface = load_surface(SURFACE, [day])
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
    day, departures, liftoff, own = args
    surface = load_surface(SURFACE, [day])
    covered = set(surface["airport"].unique()) if len(surface) else set()
    dep = departures[departures["ADEP_mvt"].isin(covered)]
    if dep.empty:
        return pd.DataFrame(columns=[ID, *CONGESTION_NAMES])
    return surface_congestion(dep, surface, liftoff, own_icao=own)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stands = pd.read_parquet(MATCHES / "stands.parquet")
    for month in args.months:
        raw = pd.read_parquet(raw_path(month), columns=DEPARTURE_COLUMNS + ["RUNWAY_mvt"])
        dep = raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
        dep["takeoff"] = (dep[TIME] - EPOCH).dt.total_seconds()
        dep["day"] = dep[TIME].dt.strftime("%Y%m%d")
        matches = pd.read_parquet(MATCHES / f"matches2_{month}.parquet").merge(dep[[ID, "RUNWAY_mvt"]], on=ID, how="inner")
        matches["day"] = matches["day"].astype(str).str.replace("-", "")
        days = sorted(dep["day"].unique())
        with Pool(args.workers) as pool:
            first = pool.map(pass_one, [(day, matches[matches["day"].eq(day)].reset_index(drop=True)) for day in days])
        timing = pd.concat([t for t, _ in first], ignore_index=True)
        points = pd.concat([p for _, p in first], ignore_index=True)
        liftoff = points.groupby(["airport", "runway"]).agg(lat=("lat", "median"), lon=("lon", "median"), n=("lat", "size")).reset_index()
        liftoff = liftoff[liftoff["n"] >= 20].reset_index(drop=True)
        own = dep[[ID]].merge(matches[[ID, "icao"]], on=ID, how="left")["icao"]
        own.index = dep.index
        with Pool(args.workers) as pool:
            second = pool.map(pass_two, [(day, dep[dep["day"].eq(day)], liftoff, own[dep["day"].eq(day)]) for day in days])
        congestion = pd.concat(second, ignore_index=True)
        positions = impute_stands(stands, dep)
        stand = stand_features(matches, positions)
        out = dep[[ID]].merge(timing, on=ID, how="left").merge(stand, on=ID, how="left").merge(congestion, on=ID, how="left")
        out = out[[ID, *FEATURE_NAMES_V19]]
        out.to_parquet(OUTPUT / f"features_v19_{month}.parquet", index=False)
        coverage = {name: round(float(out[name].notna().mean()), 3) for name in FEATURE_NAMES_V19}
        print(month, "departures", len(dep), "matched", len(matches), "lift-off points", len(liftoff), "imputed stands", int(positions["imputed"].sum()), "coverage", coverage, flush=True)


if __name__ == "__main__":
    main()
