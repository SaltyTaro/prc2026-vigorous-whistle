"""Label-free coverage of same-flight NM clocks elsewhere in the supplied movements."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TIME, TARGET, BLOCK

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [ID, TIME, "FLIGHT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AOBT_3_flt", "LOBT_flt"]
ALLOWED = [f"2025-{month:02}" for month in range(1, 12)] + ["2026-01", "2026-07"]


def main():
    if TARGET in COLUMNS or BLOCK in COLUMNS:
        raise ValueError("This audit must not read labels or airport block times.")
    destination = ROOT/"docs/ladder/clock_context_coverage.json"
    if destination.exists():
        raise ValueError("Preserve the original coverage audit.")
    inventory = read_json(ROOT/"docs/competition/download_manifest.json")["objects"]
    parts, hashes = [], {}
    for item in inventory:
        if item["key"] == "submitting.parquet":
            continue
        path = ROOT/"data/raw"/item["key"]
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise ValueError("Original supplied data changed.")
        hashes[item["key"]] = digest
        frame = pd.read_parquet(path, columns=COLUMNS)
        frame = frame.loc[frame[TIME].dt.strftime("%Y-%m").isin(ALLOWED)].copy()
        frame["movement_month"] = frame[TIME].dt.strftime("%Y-%m")
        frame["movement_year"] = frame[TIME].dt.year
        frame["has_clock"] = frame["AOBT_3_flt"].notna() | frame["LOBT_flt"].notna()
        parts.append(frame)
    pool = pd.concat(parts, ignore_index=True)
    departures = pool.loc[pool["PHASE_mvt"].eq("DEP")].copy()
    missing = departures.loc[~departures["has_clock"]].copy()
    keys = ["movement_year", "FLIGHT_ID_mvt"]
    peer_presence = pool.groupby(keys, dropna=True, observed=True)["has_clock"].any().rename("same_flight_clock_present").reset_index()
    missing = missing.merge(peer_presence, on=keys, how="left", validate="many_to_one")
    missing["same_flight_clock_present"] = missing["same_flight_clock_present"].fillna(False).astype(bool)
    potential = missing.loc[missing["same_flight_clock_present"]].copy()
    nearby = pd.DataFrame()
    if len(potential):
        candidates = pool.loc[pool["has_clock"] & pool["FLIGHT_ID_mvt"].isin(potential["FLIGHT_ID_mvt"])]
        pairs = potential.merge(candidates, on=keys, how="inner", suffixes=("_query", "_peer"), validate="many_to_many")
        near = (pairs[TIME+"_peer"]-pairs[TIME+"_query"]).dt.total_seconds().abs().le(48*3600)
        origin = pairs["ADEP_mvt_query"].eq(pairs["ADEP_flt_peer"])
        nearby = pairs.loc[near & origin].copy()
    summary = {}
    for period, mask in [("training_2025_jan_nov", departures["movement_year"].eq(2025)),
                         ("ranking_2026", departures["movement_year"].eq(2026))]:
        cohort = departures.loc[mask]
        absent = missing.loc[missing["movement_year"].eq(2025 if period.startswith("training") else 2026)]
        candidates = potential.loc[potential["movement_year"].eq(2025 if period.startswith("training") else 2026)]
        recovered = 0
        if len(nearby):
            recovered = nearby.loc[nearby["movement_year"].eq(2025 if period.startswith("training") else 2026), ID+"_query"].nunique()
        summary[period] = {"departure_rows": len(cohort), "missing_clock_rows": len(absent),
                           "same_flight_peer_has_clock_rows": len(candidates),
                           "matching_origin_peer_within_48h_rows": int(recovered)}
    if summary["ranking_2026"]["departure_rows"] != 344841:
        raise ValueError("The ranking departure population changed.")
    groups = []
    for (month, airport), group in missing.groupby(["movement_month", "ADEP_mvt"], dropna=False, observed=True):
        groups.append({"movement_month": str(month), "airport": str(airport), "missing_rows": len(group),
                       "same_flight_peer_has_clock_rows": int(group["same_flight_clock_present"].sum())})
    report = {"created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)),
              "input_sha256": hashes, "input_columns": COLUMNS, "movement_months": ALLOWED,
              "labels_read": False, "airport_block_times_read": False, "audit_used": False,
              "models_fitted": 0, "raw_identifiers_used_only_for_matching": True,
              "eligible_movement_rows": len(pool), "periods": summary, "missing_by_airport_month": groups,
              "interpretation": "Covariate coverage only. Same-flight matching requires additional consistency review before any new feature is declared. No existing feature or prediction is changed."}
    write_json(destination, report)
    print(summary)


if __name__ == "__main__":
    main()
