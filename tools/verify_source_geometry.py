"""Reconstruct the diagnostic evidence and check immutable competition artifacts."""

from pathlib import Path
import argparse
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026_source_geometry.common import CONFIG, checked_config, load_fitting, read_geometry_covariates
from prc2026_source_geometry.geometry_audit import clean_ref, coverage
from prc2026_source_geometry.geometry_review import REVIEW_CONFIG, load_review_snapshot, reviewed_map
from prc2026_source_geometry.endpoint_review import ENDPOINT_CONFIG, add_documented_endpoints, explicit_controls


def check_hash(name, expected):
    if sha256_file(ROOT / name) != expected:
        raise ValueError("Artifact hash differs: " + name)


def same(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Reconstruction differs: " + label)


def verify():
    config = checked_config(ROOT)
    review = read_json(ROOT / REVIEW_CONFIG)
    endpoint_config = read_json(ROOT / ENDPOINT_CONFIG)
    for declaration, key in [(review, "input_receipts_sha256"), (endpoint_config, "input_sha256")]:
        for name, digest in declaration[key].items():
            check_hash(name, digest)
    names = ["fitting_source_audit", "current_geometry_coverage", "geometry_coverage",
             "current_geometry_mapping_review", "geometry_mapping_review", "geometry_endpoint_review", "audit_decision"]
    reports = {name: read_json(ROOT / f"docs/source_geometry/{name}.json") for name in names}
    for name, report in reports.items():
        for path, digest in report.get("code_sha256", {}).items():
            check_hash(path, digest)
        for path, digest in report.get("input_sha256", {}).items():
            check_hash(path, digest)
        if "location_details_sha256" in report:
            check_hash(f"docs/source_geometry/{name}_locations.json", report["location_details_sha256"])
        if report["audit_used"] or report["models_fitted"] or report["predictions_changed"]:
            raise ValueError("Audit scope changed.")
    cache, failures = [], []
    for path in sorted((ROOT / "docs/source_geometry/sources").glob("*.receipt.json")):
        receipt = read_json(path)
        check_hash(receipt["path"], receipt["sha256"])
        if (ROOT / receipt["path"]).stat().st_size != receipt["size_bytes"]:
            raise ValueError("Public source size differs.")
        cache.append({"path": receipt["path"], "sha256": receipt["sha256"], "size_bytes": receipt["size_bytes"],
            "receipt": path.relative_to(ROOT).as_posix(), "receipt_sha256": sha256_file(path),
            "url": receipt["url"], "requested_snapshot": receipt["requested_snapshot"],
            "content_license": "ODbL-1.0" if receipt["query"] else
                ("CC-BY-SA-2.0" if "parking_position_definition" in receipt["path"] else "Archived license/copyright source text; map-data license is ODbL-1.0.")})
    for path in sorted((ROOT / "docs/source_geometry/sources").glob("*.failed_*.json")):
        failures.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)})
    if len(cache) != 13 or len(failures) != 3:
        raise ValueError("The frozen public source inventory changed.")
    source = reports["fitting_source_audit"]
    raw, fitting_receipt = load_fitting(ROOT, config, with_labels=True)
    same(fitting_receipt, source["population"], "fitting population")
    check_hash("docs/source_geometry/stand_month_recurrence.csv", source["recurrence_table_sha256"])
    totals = {}
    for name, groups in source["groups"].items():
        if name.endswith("_stand"):
            continue
        totals[name] = sum(item["rows"] for item in groups)
        if totals[name] != len(raw):
            raise ValueError("An aggregate lost fitting rows.")
    block = raw[TIME] - pd.to_timedelta(raw[TARGET], unit="s")
    has_clock = raw.AOBT_3_flt.notna() | raw.LOBT_flt.notna()
    schedule_difference = (raw.SCHED_TIME_UTC_mvt - block).dt.total_seconds()
    aobt_difference = (raw.AOBT_3_flt - block).dt.total_seconds()
    lobt_difference = (raw.LOBT_flt - block).dt.total_seconds()
    day_nearest = np.rint(schedule_difference / 86400)
    hour_nearest = np.rint(aobt_difference / 3600)
    flags = {"block_near_schedule": schedule_difference.abs().le(60),
        "block_near_aobt": aobt_difference.abs().le(60), "block_near_lobt": lobt_difference.abs().le(60),
        "schedule_day_multiple": day_nearest.ne(0) & (schedule_difference - day_nearest * 86400).abs().le(120),
        "nm_hour_multiple": hour_nearest.ne(0) & (aobt_difference - hour_nearest * 3600).abs().le(60),
        "target_negative": raw[TARGET].lt(0), "target_over_3h": raw[TARGET].gt(10800)}
    for row in source["groups"]["airport_nm_availability"]:
        selected = raw.ADEP_mvt.eq(row["key"]["airport"]) & has_clock.eq(row["key"]["nm_available"])
        same(int(selected.sum()), row["rows"], "airport/NM count")
        same({name: int(values[selected].sum()) for name, values in flags.items()}, row["counts"], "independent implied-block counts")
    lirf_missing = raw.ADEP_mvt.eq("LIRF") & ~has_clock
    gate = lirf_missing & raw.ADEP_flt.isna() & raw.AIRCRAFT_TYPE_mvt.isna() & raw.SCHED_TIME_UTC_mvt.notna()
    gate_check = {"missing_clock_rows": int(lirf_missing.sum()), "existing_schedule_gate_rows": int(gate.sum()),
                  "above_3h_rows_in_gate": int((gate & raw[TARGET].gt(10800)).sum()),
                  "above_3h_rows_outside_gate": int((lirf_missing & ~gate & raw[TARGET].gt(10800)).sum())}
    same(gate_check, {"missing_clock_rows":836, "existing_schedule_gate_rows":835,
                     "above_3h_rows_in_gate":102, "above_3h_rows_outside_gate":0}, "existing LIRF gate")
    del raw, block
    populations, geometry_receipt = read_geometry_covariates(ROOT, config)
    populations["ranking"] = pd.concat([populations["2026-01"], populations["2026-07"]], ignore_index=True)
    endpoint = reports["geometry_endpoint_review"]
    decision = reports["audit_decision"]
    same(geometry_receipt, endpoint["population"], "geometry covariates")
    same(geometry_receipt, decision["population"], "decision covariates")
    saved_maps = read_json(ROOT / "docs/source_geometry/geometry_endpoint_review_locations.json")
    saved_controls = read_json(ROOT / "docs/source_geometry/parking_endpoint_controls.json")
    check_hash("docs/source_geometry/parking_endpoint_controls.json", endpoint["parking_controls_sha256"])
    graph_checks = 0
    for airport in config["airports"]:
        snapshots, controls = {}, {}
        for stamp in config["geometry"]["requested_snapshots"]:
            elements, receipt = load_review_snapshot(ROOT, airport, stamp)
            snapshots[stamp] = elements
            same(receipt, endpoint["source_receipts"][f"{airport}/{stamp}"], "map source receipt")
            for item in explicit_controls(elements, endpoint_config["node_to_endpoint_tolerance_m"]):
                controls[json.dumps(item, sort_keys=True)] = item
        same(list(controls.values()), saved_controls[airport], "parking endpoint controls")
        states = list(controls.values())
        permitted = len({x["way_id"] for x in states}) >= endpoint_config["minimum_distinct_control_ways_per_airport"] and sum(x["last_node_convention_passes"] for x in states) / len(states) >= endpoint_config["minimum_last_node_control_fraction"]
        same(permitted, endpoint["airports"][airport]["endpoint_inference_permitted"], "endpoint decision")
        for stamp, elements in snapshots.items():
            expanded, inferred, conflicts = add_documented_endpoints(elements) if permitted else (elements, {}, [])
            mapped = reviewed_map(expanded, review)
            for ref, ways in inferred.items():
                if ref in mapped["parking"]:
                    mapped["parking"][ref]["position_method"] = "Inferred from the documented last node of a parking way; this node was not explicitly tagged as a parking position in the raw snapshot."
                    mapped["parking"][ref]["inferred_endpoint_source_way_ids"] = ways
            mapped["inferred_endpoint_refs"] = sorted(inferred)
            mapped["endpoint_conflicts"] = conflicts
            same(mapped, saved_maps[f"{airport}/{stamp}"], "full graph and parking records")
            supported = {ref: item for ref, item in mapped["parking"].items() if item["sources"] and all(part["method"] == "node" for part in item["sources"])}
            month = "ranking" if stamp == "current" else stamp[:7]
            rows = populations[month].loc[populations[month].ADEP_mvt.eq(airport)]
            measured = coverage(rows, {**mapped, "parking": supported})
            saved = decision["geometry"][airport]["snapshots"][stamp]
            same(measured["exact_unambiguous_parking_rows"], saved["supported_parking_rows"], "supported parking count")
            same(measured["parking_to_recorded_runway_connected_rows"], saved["supported_connected_rows"], "supported route count")
            same(measured["minimum_route_distance_quantiles_m"], saved["minimum_route_distance_quantiles_m"], "route distribution")
            graph_checks += 1
    prior = read_json(ROOT / "docs/next_lever/preservation_final.json")
    for name, digest in prior["original_input_hashes"].items():
        check_hash("data/raw/" + name, digest)
    exports = {**prior["previous_submission_hashes"], "v7": prior["new_local_export"]["sha256"]}
    for version, digest in exports.items():
        check_hash(f"runs/submissions/vigorous-whistle_{version}.parquet", digest)
    official = read_json(ROOT / "docs/surface_refit/submission_v7_official.json")
    for name, digest in official["artifact_sha256"].items():
        check_hash(name, digest)
    archive = read_json(ROOT / "docs/next_lever/final_source_package.json")
    check_hash(archive["archive"], archive["sha256"])
    permitted_narratives = {x["path"]: x["after_sha256"] for x in read_json(ROOT / "docs/surface_refit/submission_documentation_20260914.json")["updated_narratives"]}
    for item in read_json(ROOT / "docs/source_geometry/narrative_updates.json")["files"]:
        check_hash(item["snapshot"], item["before_sha256"])
        if item["before_sha256"] != permitted_narratives[item["path"]]:
            raise ValueError("Prior narrative state differs.")
        permitted_narratives[item["path"]] = item["after_sha256"]
    for name, digest in archive["files"].items():
        check_hash(name, permitted_narratives.get(name, digest))
    xml = ET.parse(ROOT / "docs/source_geometry/tests_endpoint_final_local.xml")
    test_cases = xml.findall(".//testcase")
    if len(test_cases) != 34 or xml.findall(".//failure") or xml.findall(".//error") or xml.findall(".//skipped"):
        raise ValueError("Focused test evidence does not match the completed test set.")
    inventory = {"sources": cache, "failed_attempts": failures, "map_data_attribution": "Copyright OpenStreetMap contributors, ODbL 1.0",
                 "wiki_documentation_license": "CC BY-SA 2.0, linked in the archived parking-tag guide footer.",
                 "total_source_bytes": sum(x["size_bytes"] for x in cache)}
    report = {"created_at_utc": utc_now(), "passed": True, "models_fitted": 0, "predictions_changed": False,
        "audit_used": False, "azure_actions_performed": False, "fitting_rows_verified": fitting_receipt["focal_airport_fitting_rows"],
        "complete_source_group_totals": totals, "independent_lirf_gate_check": gate_check,
        "full_geometry_maps_reconstructed": graph_checks, "public_sources_verified": len(cache),
        "focused_local_tests_passed": len(test_cases), "raw_input_hashes_verified": len(prior["original_input_hashes"]),
        "submission_versions_unchanged": list(exports), "prior_archive_files_checked": len(archive["files"]),
        "prior_frozen_archive_files_unchanged_except_five_previously_documented_narratives": len(archive["files"]) - len(permitted_narratives),
        "official_v7_artifact_hashes_checked": len(official["artifact_sha256"]),
        "input_sha256": {f"docs/source_geometry/{name}.json": sha256_file(ROOT / f"docs/source_geometry/{name}.json") for name in names},
        "code_sha256": sha256_file(Path(__file__)),
        "scope_note": "Source statistics reconstruct only common fitting labels. Geometry reads six label-free columns. All original Parquet files, including December, are byte-hashed for preservation; reserved labels are not decoded, selected, trained or scored."}
    return report, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    for path in [args.output, args.inventory]:
        if path and path.exists():
            raise ValueError("Preserve existing verification output: " + str(path))
    report, inventory = verify()
    if args.inventory:
        write_json(args.inventory, inventory)
    if args.output:
        write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, (dict, list))}, indent=2))


if __name__ == "__main__":
    main()
