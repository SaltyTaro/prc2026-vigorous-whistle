"""Summarize completed audits and separate parking points from line centroids."""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_source_geometry.common import checked_config, read_geometry_covariates
from prc2026_source_geometry.geometry_audit import clean_ref, coverage, distance_m


def main():
    output = ROOT / "docs/source_geometry/audit_decision.json"
    if output.exists():
        raise ValueError("Preserve the original audit decision.")
    config = checked_config(ROOT)
    paths = ["docs/source_geometry/fitting_source_audit.json",
             "docs/source_geometry/geometry_endpoint_review.json",
             "docs/source_geometry/geometry_endpoint_review_locations.json"]
    source, endpoint, maps = [read_json(ROOT / name) for name in paths]
    if sha256_file(ROOT / paths[2]) != endpoint["location_details_sha256"]:
        raise ValueError("Endpoint location details changed.")
    for report in [source, endpoint]:
        for name, expected in report["code_sha256"].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError("An audit producer changed: " + name)
    populations, population_receipt = read_geometry_covariates(ROOT, config)
    populations["ranking"] = pd.concat([populations["2026-01"], populations["2026-07"]], ignore_index=True)
    airports = {}
    for airport in config["airports"]:
        snapshots, historical_supported = {}, []
        for stamp in config["geometry"]["requested_snapshots"]:
            mapped = maps[f"{airport}/{stamp}"]
            # Way centroids remain in the earlier reports but are not parking points.
            supported = {ref: item for ref, item in mapped["parking"].items()
                         if item["sources"] and all(part["method"] == "node" for part in item["sources"])}
            inferred = {ref for ref, item in supported.items() if item.get("inferred_endpoint_source_way_ids")}
            if inferred and not endpoint["airports"][airport]["endpoint_inference_permitted"]:
                raise ValueError("An unapproved inferred endpoint entered the summary.")
            name = "ranking" if stamp == "current" else stamp[:7]
            rows = populations[name].loc[populations[name].ADEP_mvt.eq(airport)]
            refs = rows.STAND_mvt.map(clean_ref)
            measured = coverage(rows, {**mapped, "parking": supported})
            snapshots[stamp] = {"population": name, "rows": len(rows),
                "supported_parking_refs": len(supported),
                "supported_parking_rows": measured["exact_unambiguous_parking_rows"],
                "supported_parking_fraction": measured["exact_unambiguous_parking_fraction"],
                "supported_connected_rows": measured["parking_to_recorded_runway_connected_rows"],
                "supported_connected_fraction": measured["parking_to_recorded_runway_connected_fraction"],
                "explicit_node_rows": int(refs.isin(set(supported) - inferred).sum()),
                "validated_inferred_endpoint_rows": int(refs.isin(inferred).sum()),
                "centroid_only_rows_excluded": int(refs.isin(set(mapped["parking"]) - set(supported)).sum()),
                "minimum_route_distance_quantiles_m": measured["minimum_route_distance_quantiles_m"]}
            if stamp != "current":
                historical_supported.append(supported)
        shared = set.intersection(*[set(item) for item in historical_supported])
        stable = []
        for ref in sorted(shared):
            points = [(item[ref]["lat"], item[ref]["lon"]) for item in historical_supported]
            if max(distance_m(a, b) for a in points for b in points) <= 30:
                stable.append(ref)
        ranking = populations["ranking"].loc[populations["ranking"].ADEP_mvt.eq(airport)]
        stable_rows = int(ranking.STAND_mvt.map(clean_ref).isin(stable).sum())
        controls = endpoint["airports"][airport]
        airports[airport] = {"snapshots": snapshots,
            "endpoint_control_distinct_ways": controls["control_distinct_ways"],
            "endpoint_control_states": controls["control_states"],
            "endpoint_control_fraction": controls["last_node_control_fraction"],
            "endpoint_inference_permitted": controls["endpoint_inference_permitted"],
            "supported_stable_refs_across_four_period_starts": stable,
            "ranking_rows_with_supported_stable_ref": stable_rows,
            "ranking_fraction_with_supported_stable_ref": stable_rows / len(ranking)}
    source_summary = {"focal_fitting_rows": source["population"]["focal_airport_fitting_rows"],
        "missing_nm_groups": [{"key": item["key"], "rows": item["rows"], "counts": item["counts"]}
            for item in source["groups"]["airport_nm_availability"] if not item["key"]["nm_available"]],
        "lirf_missing_nm_aircraft_groups": [{"key": item["key"], "rows": item["rows"], "counts": item["counts"]}
            for item in source["groups"]["airport_nm_availability_aircraft_missing"]
            if item["key"]["airport"] == "LIRF" and not item["key"]["nm_available"]],
        "nonzero_day_multiple_rows": sum(item["counts"]["schedule_day_multiple"] for item in source["groups"]["airport"]),
        "missing_nm_stand_groups_meeting_declared_recurrence": sum(
            not item["key"]["nm_available"] and item["rows"] >= config["source_audit"]["minimum_stand_group_rows"]
            and item["active_months"] >= config["source_audit"]["minimum_stand_active_months"]
            for item in source["groups"]["airport_nm_availability_stand"])}
    report = {"created_at_utc": utc_now(), "status": "audits_complete",
        "input_sha256": {name: sha256_file(ROOT / name) for name in paths},
        "code_sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha256_file(Path(__file__)),
            "src/prc2026_source_geometry/common.py": sha256_file(ROOT / "src/prc2026_source_geometry/common.py"),
            "src/prc2026_source_geometry/geometry_audit.py": sha256_file(ROOT / "src/prc2026_source_geometry/geometry_audit.py")},
        "population": population_receipt, "source_reporting": source_summary, "geometry": airports,
        "recommended_next_experiment": "Declare a LIRF geometry feature comparison against v7, then add route-sharing queue context only as a separately declared candidate. Use supported historical parking points, missingness flags and unchanged fallback for unavailable geometry; score both complete original folds. LFPG requires historically credible parking points before a broad deployment of this feature.",
        "clock_decision": "No additional clock repair or broader schedule gate is supported by this fitting-only audit.",
        "source_limitations": ["OSM edit history records map state, not the date a physical airport layout became operational.",
            "Routes are undirected shortest mapped distances to any access on the named runway, not actual clearances or direction-specific thresholds.",
            "LFPG line centroids are excluded from supported parking-point coverage; failed endpoint controls were not relaxed.",
            "Historical coverage is checked at four period starts; a future feature campaign must declare snapshot assignment for all fitting and scoring months.",
            "No additional model was fitted or scored; availability is not evidence of a score improvement."],
        "models_fitted": 0, "predictions_changed": False, "audit_used": False,
        "new_official_submission": False, "azure_actions_performed": False}
    json.dumps(report, allow_nan=False)
    write_json(output, report)
    print(json.dumps({airport: {stamp: {key: item[key] for key in ["rows", "supported_parking_fraction", "supported_connected_fraction"]}
        for stamp, item in data["snapshots"].items()} for airport, data in airports.items()}, indent=2))


if __name__ == "__main__":
    main()
