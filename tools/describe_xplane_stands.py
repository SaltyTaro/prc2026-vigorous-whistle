"""Descriptive checks after the xplane_stands_probe_v1 decision (not part of the declared rule; label-free).

1. apt.dat against OpenStreetMap: the Gateway pre_data positions against the cached current OSM parking positions of the
   source-geometry campaign (docs/source_geometry/geometry_endpoint_review_locations.json), same normalised keys, LFPG and LIRF.
   Two independent maps agreeing says the catalogue's name -> position mapping is right, whatever the ADS-B controls say.
2. Own stand against a shuffled stand: for LFPG matches (2025-11, 2025-12, 2026-01, 2026-07) with a resolved stand, the first point's
   distance to its own catalogue stand against the distance to a randomly permuted row's stand (seed 20260925); and where LFPG
   gate-class anchors sit relative to their own stand.
Writes docs/xplane_stands/descriptive_checks.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026_adsb.features import metres  # noqa: E402
from probe_xplane_stands import catalogue, lfpg_matches, normalise  # noqa: E402

OSM = ROOT / "docs/source_geometry/geometry_endpoint_review_locations.json"
OUTPUT = ROOT / "docs/xplane_stands/descriptive_checks.json"
MONTHS = ["2025-11", "2025-12", "2026-01", "2026-07"]


def against_osm(locations: dict, icao: str) -> dict:
    parking = locations[f"{icao}/current"]["parking"]
    osm = pd.DataFrame([{"key": normalise(ref), "lat": v["lat"], "lon": v["lon"]} for ref, v in parking.items()])
    cat, _ = catalogue(icao, "pre_data")
    both = osm.merge(cat, on="key", suffixes=("_osm", "_apt"))
    d = metres(both["lat_osm"].to_numpy(), both["lon_osm"].to_numpy(), both["lat_apt"].to_numpy(), both["lon_apt"].to_numpy())
    return {"osm_parking_refs": int(len(osm)), "common_keys": int(len(both)), "median_m": float(np.median(d)),
            "within_25m": float(np.mean(d <= 25)), "within_40m": float(np.mean(d <= 40)), "within_100m": float(np.mean(d <= 100)),
            "over_500m": float(np.mean(d > 500))}


def own_versus_shuffled() -> dict:
    cat, _ = catalogue("LFPG", "pre_data")
    m = pd.concat([lfpg_matches(month, cat) for month in MONTHS], ignore_index=True)
    m = m[m["cat_lat"].notna()].reset_index(drop=True)
    perm = np.random.default_rng(20260925).permutation(len(m))
    shuffled = metres(m["first_lat"].to_numpy(), m["first_lon"].to_numpy(), m["cat_lat"].to_numpy()[perm], m["cat_lon"].to_numpy()[perm])
    gate = m[m["gate"]]
    anchor = metres(gate["anchor_lat"].to_numpy(), gate["anchor_lon"].to_numpy(), gate["cat_lat"].to_numpy(), gate["cat_lon"].to_numpy())
    return {"months": MONTHS, "rows": int(len(m)), "own_median_m": float(np.median(m["d"])), "shuffled_median_m": float(np.median(shuffled)),
            "own_closer_share": float(np.mean(m["d"].to_numpy() < shuffled)), "gate_rows": int(len(gate)),
            "gate_anchor_own_median_m": float(np.median(anchor))}


def main():
    locations = read_json(OSM)
    result = {"created_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)), "osm_locations_sha256": sha256_file(OSM),
              "note": "descriptive, after the declared decision; changes nothing in it",
              "apt_dat_vs_osm": {icao: against_osm(locations, icao) for icao in ["LFPG", "LIRF"]},
              "lfpg_own_vs_shuffled": own_versus_shuffled()}
    write_json(OUTPUT, result)
    print(result)


if __name__ == "__main__":
    main()
