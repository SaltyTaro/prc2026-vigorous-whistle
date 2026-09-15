"""Resolve same-ref parking nodes and their explicitly linked lead-in ways."""

from collections import defaultdict
from pathlib import Path
import json
import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from .common import CONFIG, checked_config, read_geometry_covariates
from .geometry_audit import clean_ref, coverage, distance_m, graph_from_elements, position_index, representative

REVIEW_CONFIG = "configs/source_geometry_mapping_review_v1.json"


def resolved_positions(elements, max_span_m):
    groups = defaultdict(list)
    for element in elements:
        tags = element.get("tags", {})
        ref = clean_ref(tags.get("ref"))
        if tags.get("aeroway") == "parking_position" and ref:
            groups[ref].append(element)
    valid, ambiguous = {}, {}
    for ref, items in sorted(groups.items()):
        nodes = {item["id"]: item for item in items if item["type"] == "node" and "lat" in item and "lon" in item}
        points, sources, linked = [], [], []
        for item in items:
            shared = sorted(set(item.get("nodes", [])).intersection(nodes)) if item["type"] == "way" else []
            if shared:
                linked.append({"way": item["id"], "same_ref_parking_node_members": shared})
                continue
            point, method = representative(item)
            if point is not None:
                points.append(point)
                sources.append({"element": f'{item["type"]}/{item["id"]}', "lat": point[0], "lon": point[1], "method": method})
        if not points:
            continue
        span = max(distance_m(a, b) for a in points for b in points)
        entry = {"lat": float(np.mean([p[0] for p in points])), "lon": float(np.mean([p[1] for p in points])),
                 "span_m": span, "sources": sources, "linked_same_ref_ways": linked,
                 "named_parking_node_ids": sorted(nodes),
                 "position_method": "Explicit same-ref node membership resolves lead-in geometry; unlinked representatives retain the 30m check."}
        (valid if span <= max_span_m else ambiguous)[ref] = entry
    return valid, ambiguous


def reviewed_map(elements, config):
    geo = config["geometry"]
    parking, ambiguous = resolved_positions(elements, geo["max_duplicate_position_span_m"])
    gates, ambiguous_gates = position_index(elements, "gate", geo["max_duplicate_position_span_m"])
    valid_nodes = {node for entry in parking.values() for node in entry["named_parking_node_ids"]}
    graph_elements, added_ways = [], []
    for item in elements:
        tags = item.get("tags", {})
        ids = item.get("nodes", [])
        if item["type"] == "way" and tags.get("aeroway") == "parking_position":
            if tags.get("area") != "yes" and len(ids) > 1 and ids[0] != ids[-1] and valid_nodes.intersection(ids):
                # Reuse the frozen line-graph builder; the original source remains unchanged.
                graph_elements.append({**item, "tags": {**tags, "aeroway": "taxilane"}})
                added_ways.append(item["id"])
        else:
            graph_elements.append(item)
    graph, coordinates, accesses = graph_from_elements(graph_elements)
    components = list(nx.connected_components(graph))
    component_of = {node: number for number, nodes in enumerate(components) for node in nodes}
    distances = {name: nx.multi_source_dijkstra_path_length(graph, nodes, weight="weight") if nodes else {}
                 for name, nodes in accesses.items()}
    node_ids = list(graph.nodes)
    if node_ids:
        lat0 = math.radians(float(np.mean([coordinates[node][0] for node in node_ids])))
        project = lambda p: (6371008.8 * math.radians(p[1]) * math.cos(lat0), 6371008.8 * math.radians(p[0]))
        tree = cKDTree([project(coordinates[node]) for node in node_ids])
        for record in parking.values():
            _, index = tree.query(project((record["lat"], record["lon"])))
            node = node_ids[int(index)]
            snap = distance_m((record["lat"], record["lon"]), coordinates[node])
            accepted = snap <= geo["max_graph_snap_distance_m"]
            record.update(nearest_node=node, graph_snap_distance_m=snap, graph_snap_accepted=accepted,
                graph_component=component_of[node], runway_minimum_distance_m={name: float(values[node] + snap)
                    for name, values in distances.items() if accepted and node in values})
    else:
        for record in parking.values():
            record.update(graph_snap_accepted=False, runway_minimum_distance_m={})
    return {"parking": parking, "ambiguous_parking": ambiguous, "gates": gates, "ambiguous_gates": ambiguous_gates,
        "parking_refs_with_explicit_linked_ways": sum(bool(item["linked_same_ref_ways"]) for item in parking.values()),
        "parking_lead_in_way_ids_added_to_graph": added_ways,
        "graph": {"nodes": len(node_ids), "edges": graph.number_of_edges(), "connected_components": len(components),
            "largest_component_nodes": max(map(len, components), default=0),
            "runway_access_node_counts": {name: len(values) for name, values in sorted(accesses.items())}},
        "route_interpretation": geo["route_proxy"]}


def load_review_snapshot(root, airport, stamp):
    label = "current" if stamp == "current" else stamp[:10].replace("-", "")
    receipt = read_json(root / "docs/source_geometry/sources" / f"{airport}_{label}.json.receipt.json")
    permitted = {sha256_file(root / CONFIG), sha256_file(root / REVIEW_CONFIG)}
    if receipt["declaration_sha256"] not in permitted or receipt["requested_snapshot"] != stamp:
        raise ValueError("Source belongs to an undeclared snapshot.")
    path = root / receipt["path"]
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("Geometry source hash differs.")
    elements = read_json(path)["elements"]
    if stamp != "current" and any(item.get("timestamp", "") > stamp for item in elements):
        raise ValueError("Historical geometry contains a later OSM edit.")
    return elements, receipt


def main(root, *, complete):
    output = root / "docs/source_geometry" / ("geometry_mapping_review.json" if complete else "current_geometry_mapping_review.json")
    if output.exists():
        raise ValueError("Preserve the existing mapping review.")
    checked_config(root)
    config = read_json(root / REVIEW_CONFIG)
    for name, expected in config["input_receipts_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError("Mapping-review parent changed: " + name)
    populations, population_receipt = read_geometry_covariates(root, config)
    populations["ranking"] = pd.concat([populations["2026-01"], populations["2026-07"]], ignore_index=True)
    airports, maps, sources = {}, {}, {}
    for airport in config["airports"]:
        elements, receipt = load_review_snapshot(root, airport, "current")
        mapped = reviewed_map(elements, config)
        current = {name: coverage(frame.loc[frame.ADEP_mvt.eq(airport)], mapped) for name, frame in populations.items()}
        eligible = current["ranking"]["exact_unambiguous_parking_fraction"] >= config["geometry"]["historical_fetch_minimum_current_exact_parking_row_coverage"]
        airports[airport] = {"current": current, "current_graph": mapped["graph"],
            "parking_refs_resolved_by_node_membership": mapped["parking_refs_with_explicit_linked_ways"],
            "linked_parking_ways_in_graph": len(mapped["parking_lead_in_way_ids_added_to_graph"]),
            "eligible_for_declared_historical_fetch": eligible, "historical": {}}
        maps[f"{airport}/current"], sources[f"{airport}/current"] = mapped, receipt
        if complete and eligible:
            historical = []
            for stamp in config["geometry"]["requested_snapshots"][1:]:
                elements, receipt = load_review_snapshot(root, airport, stamp)
                old = reviewed_map(elements, config)
                frame = populations[stamp[:7]].loc[populations[stamp[:7]].ADEP_mvt.eq(airport)]
                airports[airport]["historical"][stamp] = {"matching_month": coverage(frame, old), "graph": old["graph"]}
                maps[f"{airport}/{stamp}"], sources[f"{airport}/{stamp}"] = old, receipt
                historical.append(old)
            shared = set.intersection(*[set(item["parking"]) for item in historical])
            stable, moving = [], []
            for ref in sorted(shared):
                points = [(item["parking"][ref]["lat"], item["parking"][ref]["lon"]) for item in historical]
                span = max(distance_m(a, b) for a in points for b in points)
                (stable if span <= config["geometry"]["max_duplicate_position_span_m"] else moving).append(ref)
            rank = populations["ranking"].loc[populations["ranking"].ADEP_mvt.eq(airport)]
            count = int(rank.STAND_mvt.map(clean_ref).isin(stable).sum())
            airports[airport]["historical_position_stability"] = {"shared_refs": len(shared), "stable_refs": stable,
                "moved_more_than_30m_refs": moving, "ranking_rows_with_stable_ref": count,
                "ranking_fraction_with_stable_ref": count / len(rank),
                "interpretation": "Recorded OSM state at period starts; not proof of physical layout dates or within-month stability."}
    details = output.with_name(output.stem + "_locations.json")
    if details.exists():
        raise ValueError("Preserve existing resolved locations.")
    write_json(details, maps)
    code = [Path(__file__), Path(__file__).with_name("geometry_audit.py"), Path(__file__).with_name("common.py"),
            root / "tools/review_airport_geometry.py"]
    report = {"created_at_utc": utc_now(), "complete_historical_check": complete,
        "declaration_sha256": sha256_file(root / REVIEW_CONFIG), "population": population_receipt,
        "airports": airports, "source_receipts": sources, "location_details_sha256": sha256_file(details),
        "code_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in code},
        "audit_used": False, "labels_read": False, "models_fitted": 0, "predictions_changed": False,
        "interpretation": "Explicit OSM node membership resolves duplicate representations; no fuzzy alias or gate substitution. Initial centroid-based reports remain preserved. Geometry availability does not establish model gain."}
    json.dumps(report, allow_nan=False)
    write_json(output, report)
    print(json.dumps({"complete": complete, "airports": {name: {
        "ranking_rows": item["current"]["ranking"]["rows"],
        "current_parking_fraction": item["current"]["ranking"]["exact_unambiguous_parking_fraction"],
        "current_connected_fraction": item["current"]["ranking"]["parking_to_recorded_runway_connected_fraction"],
        "qualified_for_history": item["eligible_for_declared_historical_fetch"],
        "historical": {stamp: {k: data["matching_month"][k] for k in ["rows", "exact_unambiguous_parking_fraction", "parking_to_recorded_runway_connected_fraction"]}
                       for stamp, data in item["historical"].items()},
        "stable_ranking_fraction": item.get("historical_position_stability", {}).get("ranking_fraction_with_stable_ref")}
        for name, item in airports.items()}}, indent=2))
