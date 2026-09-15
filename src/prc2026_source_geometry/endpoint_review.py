"""Validate parking-way endpoints against explicit OSM parking-node controls."""

from pathlib import Path
import json
import pandas as pd

from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from .common import checked_config, read_geometry_covariates
from .geometry_audit import clean_ref, coverage, distance_m
from .geometry_review import REVIEW_CONFIG, load_review_snapshot, reviewed_map

ENDPOINT_CONFIG = "configs/source_geometry_endpoint_review_v1.json"


def explicit_controls(elements, tolerance):
    nodes = {item["id"]: item for item in elements if item["type"] == "node"
             and item.get("tags", {}).get("aeroway") == "parking_position"}
    records = []
    for item in elements:
        tags = item.get("tags", {})
        ids, geometry = item.get("nodes", []), item.get("geometry", [])
        if item["type"] != "way" or tags.get("aeroway") != "parking_position" or tags.get("area") == "yes" or len(ids) < 2 or ids[0] == ids[-1] or not geometry or not geometry[-1]:
            continue
        ref = clean_ref(tags.get("ref"))
        matches = [nodes[node] for node in ids if node in nodes and ref and clean_ref(nodes[node].get("tags", {}).get("ref")) == ref]
        if not matches:
            continue
        end = (geometry[-1]["lat"], geometry[-1]["lon"])
        span = max(distance_m((node["lat"], node["lon"]), end) for node in matches)
        records.append({"way_id": item["id"], "way_version": item.get("version"), "ref": ref,
            "node_controls": [{"node_id": node["id"], "lat": node["lat"], "lon": node["lon"]} for node in matches],
            "last_node": ids[-1], "maximum_control_distance_to_last_node_m": span,
            "last_node_convention_passes": span <= tolerance})
    return records


def add_documented_endpoints(elements):
    explicit = {item["id"]: item for item in elements if item["type"] == "node"
                and item.get("tags", {}).get("aeroway") == "parking_position"}
    named = {clean_ref(item.get("tags", {}).get("ref")) for item in explicit.values()}
    added, source_ways, conflicts = {}, {}, []
    for item in elements:
        tags = item.get("tags", {})
        ref = clean_ref(tags.get("ref"))
        ids, geometry = item.get("nodes", []), item.get("geometry", [])
        if item["type"] != "way" or tags.get("aeroway") != "parking_position" or not ref or ref in named:
            continue
        if tags.get("area") == "yes" or len(ids) < 2 or ids[0] == ids[-1] or not geometry or not geometry[-1]:
            continue
        final = ids[-1]
        if final in explicit and clean_ref(explicit[final].get("tags", {}).get("ref")) != ref:
            conflicts.append({"way_id": item["id"], "ref": ref, "existing_parking_ref": explicit[final].get("tags", {}).get("ref")})
            continue
        key = (ref, final)
        added[key] = {"type": "node", "id": final, "lat": geometry[-1]["lat"], "lon": geometry[-1]["lon"],
                      "tags": {"aeroway": "parking_position", "ref": ref}, "derived_from_parking_way": True}
        source_ways.setdefault(ref, []).append(item["id"])
    endpoint_refs = {}
    for ref, final in added:
        endpoint_refs.setdefault(final, set()).add(ref)
    for final, refs in endpoint_refs.items():
        if len(refs) > 1:
            conflicts.extend({"ref": ref, "endpoint_node": final, "conflicting_derived_refs": sorted(refs)} for ref in sorted(refs))
    rejected = {item["ref"] for item in conflicts}
    retained = [item for item in elements if not (item["type"] == "way"
                and item.get("tags", {}).get("aeroway") == "parking_position"
                and clean_ref(item.get("tags", {}).get("ref")) in rejected)]
    return [*retained, *[item for (ref, _), item in added.items() if ref not in rejected]], {
        ref: ways for ref, ways in source_ways.items() if ref not in rejected}, conflicts


def main(root):
    output = root / "docs/source_geometry/geometry_endpoint_review.json"
    if output.exists():
        raise ValueError("Preserve the original endpoint review.")
    checked_config(root)
    config = read_json(root / REVIEW_CONFIG)
    endpoint = read_json(root / ENDPOINT_CONFIG)
    for name, digest in endpoint["input_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("Endpoint-review parent changed: " + name)
    populations, population_receipt = read_geometry_covariates(root, config)
    populations["ranking"] = pd.concat([populations["2026-01"], populations["2026-07"]], ignore_index=True)
    airports, maps, sources, all_controls = {}, {}, {}, {}
    for airport in config["airports"]:
        snapshots, unique = {}, {}
        for stamp in config["geometry"]["requested_snapshots"]:
            elements, receipt = load_review_snapshot(root, airport, stamp)
            snapshots[stamp] = elements
            sources[f"{airport}/{stamp}"] = receipt
            for control in explicit_controls(elements, endpoint["node_to_endpoint_tolerance_m"]):
                key = json.dumps(control, sort_keys=True)
                unique[key] = control
        controls = list(unique.values())
        distinct_ways = len({record["way_id"] for record in controls})
        fraction = sum(item["last_node_convention_passes"] for item in controls) / len(controls) if controls else 0.
        permitted = distinct_ways >= endpoint["minimum_distinct_control_ways_per_airport"] and fraction >= endpoint["minimum_last_node_control_fraction"]
        airports[airport] = {"control_distinct_ways": distinct_ways, "control_states": len(controls),
            "last_node_control_fraction": fraction, "endpoint_inference_permitted": permitted,
            "control_exceptions": [item for item in controls if not item["last_node_convention_passes"]],
            "current": {}, "historical": {}}
        all_controls[airport] = controls
        historical = []
        for stamp, elements in snapshots.items():
            expanded, inferred, conflicts = add_documented_endpoints(elements) if permitted else (elements, {}, [])
            mapped = reviewed_map(expanded, config)
            for ref, ways in inferred.items():
                if ref in mapped["parking"]:
                    mapped["parking"][ref]["position_method"] = "Inferred from the documented last node of a parking way; this node was not explicitly tagged as a parking position in the raw snapshot."
                    mapped["parking"][ref]["inferred_endpoint_source_way_ids"] = ways
            mapped["inferred_endpoint_refs"] = sorted(inferred)
            mapped["endpoint_conflicts"] = conflicts
            maps[f"{airport}/{stamp}"] = mapped
            if stamp == "current":
                airports[airport]["current"] = {name: coverage(frame.loc[frame.ADEP_mvt.eq(airport)], mapped) for name, frame in populations.items()}
            else:
                frame = populations[stamp[:7]].loc[populations[stamp[:7]].ADEP_mvt.eq(airport)]
                airports[airport]["historical"][stamp] = {"matching_month": coverage(frame, mapped), "graph": mapped["graph"],
                    "inferred_endpoint_refs": len(inferred), "endpoint_conflicts": conflicts}
                historical.append(mapped)
        shared = set.intersection(*[set(item["parking"]) for item in historical])
        stable, moved = [], []
        for ref in sorted(shared):
            points = [(item["parking"][ref]["lat"], item["parking"][ref]["lon"]) for item in historical]
            (stable if max(distance_m(a, b) for a in points for b in points) <= endpoint["stability_tolerance_m"] else moved).append(ref)
        ranking = populations["ranking"].loc[populations["ranking"].ADEP_mvt.eq(airport)]
        count = int(ranking.STAND_mvt.map(clean_ref).isin(stable).sum())
        airports[airport]["historical_position_stability"] = {"shared_refs": len(shared), "stable_refs": stable,
            "moved_more_than_30m_refs": moved, "ranking_rows_with_stable_ref": count, "ranking_fraction_with_stable_ref": count / len(ranking)}
    details = output.with_name("geometry_endpoint_review_locations.json")
    controls_path = output.with_name("parking_endpoint_controls.json")
    if details.exists() or controls_path.exists():
        raise ValueError("Preserve original endpoint detail artifacts.")
    write_json(details, maps)
    write_json(controls_path, all_controls)
    code = [Path(__file__), Path(__file__).with_name("geometry_review.py"), Path(__file__).with_name("geometry_audit.py"),
            Path(__file__).with_name("common.py"), root / "tools/review_parking_endpoints.py"]
    report = {"created_at_utc": utc_now(), "complete_historical_check": True,
        "declaration_sha256": sha256_file(root / ENDPOINT_CONFIG), "population": population_receipt,
        "airports": airports, "source_receipts": sources, "location_details_sha256": sha256_file(details),
        "parking_controls_sha256": sha256_file(controls_path), "code_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in code},
        "audit_used": False, "labels_read": False, "models_fitted": 0, "predictions_changed": False,
        "interpretation": endpoint["interpretation"]}
    json.dumps(report, allow_nan=False)
    write_json(output, report)
    print(json.dumps({airport: {"controls": item["control_distinct_ways"], "last_node_fraction": item["last_node_control_fraction"],
        "endpoint_inference_permitted": item["endpoint_inference_permitted"],
        "historical": {date: {k: row["matching_month"][k] for k in ["rows", "exact_unambiguous_parking_fraction", "parking_to_recorded_runway_connected_fraction"]} for date, row in item["historical"].items()},
        "stable_ranking_fraction": item["historical_position_stability"]["ranking_fraction_with_stable_ref"]}
        for airport, item in airports.items()}, indent=2))
