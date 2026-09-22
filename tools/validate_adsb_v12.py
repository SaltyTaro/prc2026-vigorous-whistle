"""Do the extra matches carry real signal? December days, against the actual taxi-out label.

For a gate row the matcher's own pushback estimate is takeoff - motion_start, which should approximate the
label (block -> takeoff). This compares that estimate's error for the rows each fix adds or converts,
against what the frozen matcher gave the same rows. Development month only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path
sys.path.insert(0, str(ROOT / "src"))
from prc2026.features import ID, PHASE, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS as FROZEN_COLUMNS, EPOCH, load_surface
from prc2026_adsb.features import match_departures as frozen_match
from prc2026_adsb_v12.matching import DEPARTURE_COLUMNS as V12_COLUMNS, match_departures as v12_match

SURFACE = ROOT / "data/external/adsblol/surface"
DAYS = ["20251204", "20251211", "20251218", "20251225"]
CONFIGS = {"minpts1": {"min_ground_points": 1}, "radius100": {"stand_radius_m": 100.0},
           "all": {"min_ground_points": 1, "callsign_variant": True, "stand_radius_m": 100.0}}

stands = pd.read_parquet(ROOT / "data/external/adsblol/matches_v11/stands.parquet")
raw = pd.read_parquet(ROOT / "data/raw/training_2025-12-01_2026-01-01.parquet",
                      columns=sorted(set(FROZEN_COLUMNS) | set(V12_COLUMNS) | {"TAXITIME_SEC_mvt"}))
dep_all = raw[raw[PHASE].eq("DEP")].reset_index(drop=True)
dep_all["day"] = dep_all[TIME].dt.strftime("%Y%m%d")

frames = {name: [] for name in ["frozen", *CONFIGS]}
labels = []
for day in DAYS:
    dep = dep_all[dep_all["day"].eq(day)].drop(columns=["day"]).reset_index(drop=True)
    surface = load_surface(SURFACE, [day])
    if surface.empty or dep.empty:
        continue
    labels.append(dep[[ID, "TAXITIME_SEC_mvt"]].assign(takeoff=(dep[TIME] - EPOCH).dt.total_seconds()))
    frames["frozen"].append(frozen_match(dep[[c for c in FROZEN_COLUMNS if c in dep.columns]], surface, stands))
    for name, options in CONFIGS.items():
        frames[name].append(v12_match(dep, surface, stands, **options))

label = pd.concat(labels, ignore_index=True).set_index(ID)
tables = {n: pd.concat(v, ignore_index=True).set_index(ID) for n, v in frames.items() if any(len(x) for x in v)}


def estimate_error(table, ids):
    """|(takeoff - motion_start) - label| in seconds for the given ids."""
    sub = table.reindex(ids)
    est = label.loc[ids, "takeoff"].to_numpy() - sub["motion_start"].to_numpy()
    return np.abs(est - label.loc[ids, "TAXITIME_SEC_mvt"].to_numpy(dtype=float))


frozen = tables["frozen"]
print(f"December sample: {len(label)} departures, frozen matched {len(frozen)} ({100 * len(frozen) / len(label):.1f}%)")
print(f"frozen gate rows: median |pushback estimate - label| = {np.median(estimate_error(frozen, frozen.index[frozen['gate']])):.0f} s")
print(f"frozen dark rows: median |pushback estimate - label| = {np.median(estimate_error(frozen, frozen.index[~frozen['gate']])):.0f} s")
print()
for name, table in tables.items():
    if name == "frozen":
        continue
    new_ids = table.index.difference(frozen.index)
    converted = [i for i in table.index.intersection(frozen.index) if table.loc[i, "gate"] and not frozen.loc[i, "gate"]]
    print(f"=== {name}")
    if len(new_ids):
        err = estimate_error(table, new_ids)
        gate = table.loc[new_ids, "gate"].to_numpy()
        print(f"  newly matched rows: {len(new_ids)} ({100 * gate.mean():.0f}% gate)  median |estimate - label| "
              f"= {np.median(err):.0f} s   (gate {np.median(err[gate]) if gate.any() else float('nan'):.0f} s, dark {np.median(err[~gate]) if (~gate).any() else float('nan'):.0f} s)")
        print(f"  their median ground points: {table.loc[new_ids, 'points'].median():.0f}")
    if converted:
        before, after = estimate_error(frozen, converted), estimate_error(table, converted)
        print(f"  dark -> gate conversions: {len(converted)}  median |estimate - label| {np.median(before):.0f} s -> {np.median(after):.0f} s"
              f"   (better on {100 * (after < before).mean():.0f}% of them)")
        print(f"  their median first-point distance to the stand: {table.loc[converted, 'first_dist_stand_m'].median():.0f} m")
