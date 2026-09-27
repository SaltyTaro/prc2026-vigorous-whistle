"""S4: ADS-B pass-2 matches (frozen stands), the learners' 27 ADS-B features and the v19 features for a movement file's departures.

The frozen functions (prc2026_adsb.features.match_departures / departure_features / neighbour_features, prc2026_adsb.features_v19)
applied exactly as tools/build_adsb_features.py (per-day matching with a frozen stands table), tools/run_adsb_learner.py::adsb_table
(one table over every month of the file) and tools/build_adsb_v19_features.py (month-level lift-off points and stand imputation) do,
but on an arbitrary file and writing under the run root. Two match directories are used, as for the ranking: the committee learners',
networks' and tokens' `data/external/adsblol/matches` (1,332 stands) and the residual stage's `data/external/adsblol/matches_v11`.
Surface days come from data/external/adsblol/surface (extracted per day by tools/adsb_extract_day.py; new dates: `final_predict.py external`).
Label-free. Note: the neighbour features take the previous/next departure after an unstable sort, so a rebuild can move a few
dozen nb_runway_prev/next values against the frozen tables (docs/adsb_nn/README.md).
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

from prc2026.artifacts import sha256_file
from prc2026.features import ID, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS, EPOCH, FEATURE_NAMES, NEIGHBOUR_NAMES, departure_features, load_surface, match_departures, neighbour_features
from prc2026_adsb.features_v19 import FEATURE_NAMES_V19, impute_stands, stand_features

from .common import ROOT

sys.path.insert(0, str(ROOT / "tools"))
from build_adsb_v19_features import pass_one, pass_two  # noqa: E402  (the frozen per-day passes, imported unchanged)

SURFACE = ROOT / "data/external/adsblol/surface"
MATCH_DIRS = {"learners": Path("data/external/adsblol/matches"), "residual": Path("data/external/adsblol/matches_v11")}
ADSB_NAMES = [*FEATURE_NAMES, *NEIGHBOUR_NAMES]


def _one_day(args):
    day, departures, stands = args
    surface = load_surface(SURFACE, [day])
    if surface.empty or departures.empty:
        return pd.DataFrame()
    matches = match_departures(departures, surface, stands)
    matches["day"] = day
    return matches


def file_days(departures: pd.DataFrame) -> list[str]:
    return sorted(departures[TIME].dt.strftime("%Y%m%d").unique().tolist())


def surface_days_missing(departures: pd.DataFrame) -> list[str]:
    return [d for d in file_days(departures) if not (SURFACE / f"surface_{d}.parquet").exists()]


def match_month(departures: pd.DataFrame, month: str, stands: pd.DataFrame, workers: int) -> pd.DataFrame:
    """tools/build_adsb_features.py --stands for one month of the file's departures (day-parallel)."""
    dep = departures.loc[departures[TIME].dt.strftime("%Y-%m").eq(month), DEPARTURE_COLUMNS].reset_index(drop=True)
    dep["day"] = dep[TIME].dt.strftime("%Y%m%d")
    days = sorted(dep["day"].unique())
    jobs = [(day, dep[dep["day"].eq(day)].drop(columns=["day"]).reset_index(drop=True), stands) for day in days]
    with mp.Pool(max(1, min(workers, len(jobs)))) as pool:
        parts = pool.map(_one_day, jobs)
    return pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()


def learner_table(departures: pd.DataFrame, matches: pd.DataFrame, stands: pd.DataFrame) -> pd.DataFrame:
    """tools/run_adsb_learner.py::adsb_table on the file's departures (every month's matches together)."""
    dep = departures[[*DEPARTURE_COLUMNS, "RUNWAY_mvt"]].reset_index(drop=True)
    feats = departure_features(dep, matches, stands).merge(neighbour_features(dep, matches), on=ID, validate="one_to_one")
    if list(feats.columns) != [ID, *ADSB_NAMES]:
        raise ValueError("Unexpected ADS-B feature columns.")
    return feats


def v19_month(departures: pd.DataFrame, month: str, matches: pd.DataFrame, stands: pd.DataFrame, workers: int) -> pd.DataFrame:
    """tools/build_adsb_v19_features.py for one month of the file (its lift-off points and stand imputation are month-level)."""
    dep = departures.loc[departures[TIME].dt.strftime("%Y-%m").eq(month), [*DEPARTURE_COLUMNS, "RUNWAY_mvt"]].reset_index(drop=True)
    dep["takeoff"] = (dep[TIME] - EPOCH).dt.total_seconds()
    dep["day"] = dep[TIME].dt.strftime("%Y%m%d")
    matches = matches.merge(dep[[ID, "RUNWAY_mvt"]], on=ID, how="inner")
    matches["day"] = matches["day"].astype(str).str.replace("-", "")
    days = sorted(dep["day"].unique())
    with mp.Pool(max(1, min(workers, len(days)))) as pool:
        first = pool.map(pass_one, [(day, matches[matches["day"].eq(day)].reset_index(drop=True)) for day in days])
    timing = pd.concat([t for t, _ in first], ignore_index=True)
    points = pd.concat([p for _, p in first], ignore_index=True)
    liftoff = points.groupby(["airport", "runway"]).agg(lat=("lat", "median"), lon=("lon", "median"), n=("lat", "size")).reset_index()
    liftoff = liftoff[liftoff["n"] >= 20].reset_index(drop=True)
    own = dep[[ID]].merge(matches[[ID, "icao"]], on=ID, how="left")["icao"]
    own.index = dep.index
    with mp.Pool(max(1, min(workers, len(days)))) as pool:
        second = pool.map(pass_two, [(day, dep[dep["day"].eq(day)], liftoff, own[dep["day"].eq(day)]) for day in days])
    congestion = pd.concat(second, ignore_index=True)
    positions = impute_stands(stands, dep)
    stand = stand_features(matches, positions)
    out = dep[[ID]].merge(timing, on=ID, how="left").merge(stand, on=ID, how="left").merge(congestion, on=ID, how="left")
    return out[[ID, *FEATURE_NAMES_V19]]


def build(run, departures: pd.DataFrame, months: list[str], workers: int) -> dict:
    """Matches for both directories (per month), the learners' ADS-B table and the v19 table; returns the record (files written under <root>/adsb)."""
    record = {"surface_days_missing": surface_days_missing(departures), "match_dirs": {}, "outputs": []}
    matches_by = {}
    for label, directory in MATCH_DIRS.items():
        stands = pd.read_parquet(ROOT / directory / "stands.parquet")
        parts = []
        for month in months:
            target = run.path("adsb", label, f"matches2_{month}.parquet")
            if target.exists():
                matches = pd.read_parquet(target)
            else:
                matches = match_month(departures, month, stands, workers)
                matches.to_parquet(target, index=False, compression="zstd")
            parts.append(matches)
            record["outputs"].append(target)
        matches_by[label] = pd.concat(parts, ignore_index=True)
        record["match_dirs"][label] = {"directory": directory.as_posix(), "stands_sha256": sha256_file(ROOT / directory / "stands.parquet"), "stands": int(len(stands)),
                                       "matched": int(len(matches_by[label])), "gate_share": float(matches_by[label]["gate"].mean()) if len(matches_by[label]) else None}
    stands = pd.read_parquet(ROOT / MATCH_DIRS["learners"] / "stands.parquet")
    table = learner_table(departures, matches_by["learners"], stands)
    record["outputs"].append(run.path("adsb", "learner_adsb.parquet"))
    table.to_parquet(record["outputs"][-1], index=False, compression="zstd")
    parts = []
    for month in months:
        month_matches = matches_by["learners"].loc[matches_by["learners"][ID].isin(departures.loc[departures[TIME].dt.strftime("%Y-%m").eq(month), ID])]
        parts.append(v19_month(departures, month, month_matches.reset_index(drop=True), stands, workers))
    v19 = pd.concat(parts, ignore_index=True)
    record["outputs"].append(run.path("adsb", "v19.parquet"))
    v19.to_parquet(record["outputs"][-1], index=False, compression="zstd")
    record["learner_table"] = {"rows": int(len(table)), "matched": int(table["adsb_matched"].sum())}
    record["v19"] = {"rows": int(len(v19)), "coverage": {n: round(float(v19[n].notna().mean()), 4) for n in FEATURE_NAMES_V19}}
    return record
