"""Exact parking references and conservative mapped taxiway connectivity."""

from pathlib import Path
from collections import Counter, defaultdict
import json
import math
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from prc2026.artifacts import ResourceMonitor, read_json, sha256_file, utc_now, write_json
from .common import CONFIG, checked_config, read_geometry_covariates


def clean_ref(value):
    if value is None or pd.isna(value):
        return None
    return str(value).strip().upper() or None


def runway_ref(value):
    value = clean_ref(value)
    if value is None or not re.fullmatch(r"(?:0?[1-9]|[12][0-9]|3[0-6])[LRC]?", value):
        return None
    match = re.fullmatch(r"([0-9]+)([LRC]?)", value)
    return f"{int(match[1]):02}{match[2]}"


def distance_m(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371008.8 * 2 * math.asin(math.sqrt(min(1., h)))


def representative(element):
    if element["type"] == "node" and "lat" in element and "lon" in element:
        return (float(element["lat"]), float(element["lon"])), "node"
    if element["type"] == "way" and element.get("geometry"):
        points = [(item["lat"], item["lon"]) for item in element["geometry"] if item is not None]
        if points:
            return tuple(map(float, np.mean(points, axis=0))), "way_geometry_mean"
    return None, None


def position_index(elements, kind, max_span_m):
    groups = defaultdict(list)
    for element in elements:
        tags = element.get("tags", {})
        ref = clean_ref(tags.get("ref"))
        if tags.get("aeroway") != kind or not ref:
            continue
        point, method = representative(element)
        if point is not None:
            groups[ref].append({"element": f'{element["type"]}/{element["id"]}',
                                "lat": point[0], "lon": point[1], "method": method})
    valid, ambiguous = {}, {}
    for ref, items in sorted(groups.items()):
        span = max((distance_m((a["lat"], a["lon"]), (b["lat"], b["lon"]))
                    for a in items for b in items), default=0.)
        record = {"lat": float(np.mean([item["lat"] for item in items])),
                  "lon": float(np.mean([item["lon"] for item in items])),
                  "span_m": span, "sources": items}
        (valid if span <= max_span_m else ambiguous)[ref] = record
    return valid, ambiguous


def graph_from_elements(elements):
    graph, coordinates, runways = nx.Graph(), {}, defaultdict(set)
    for element in elements:
        tags = element.get("tags", {})
        if element["type"] != "way" or not element.get("nodes") or not element.get("geometry"):
            continue
        if tags.get("aeroway") not in {"taxiway", "taxilane", "runway"} or tags.get("area") == "yes":
            continue
        ids, geom = element["nodes"], element["geometry"]
        if len(ids) != len(geom):
            raise ValueError("OSM way geometry and node sequence differ.")
        points = {}
        for node, point in zip(ids, geom):
            if point is not None:
                points[node] = (float(point["lat"]), float(point["lon"]))
        if tags["aeroway"] == "runway":
            for ref in str(tags.get("ref", "")).split("/"):
                name = runway_ref(ref)
                if name:
                    runways[name].update(points)
            continue
        for node, point in points.items():
            if node in coordinates and distance_m(coordinates[node], point) > .1:
                raise ValueError("A shared taxiway node has inconsistent coordinates.")
            coordinates[node] = point
        for start, end in zip(ids[:-1], ids[1:]):
            if start not in points or end not in points or start == end:
                continue
            length = distance_m(points[start], points[end])
            if not graph.has_edge(start, end) or length < graph[start][end]["weight"]:
                graph.add_edge(start, end, weight=length)
    accesses = {name: nodes.intersection(graph.nodes) for name, nodes in runways.items()}
    return graph, coordinates, accesses


def map_snapshot(elements, config):
    geo = config["geometry"]
    parking, ambiguous = position_index(elements, "parking_position", geo["max_duplicate_position_span_m"])
    gates, gate_ambiguous = position_index(elements, "gate", geo["max_duplicate_position_span_m"])
    graph, coordinates, accesses = graph_from_elements(elements)
    components = list(nx.connected_components(graph))
    component_of = {node: number for number, part in enumerate(components) for node in part}
    distances = {name: nx.multi_source_dijkstra_path_length(graph, nodes, weight="weight") if nodes else {}
                 for name, nodes in accesses.items()}
    nodes = list(graph.nodes)
    if nodes:
        lat0 = math.radians(float(np.mean([coordinates[node][0] for node in nodes])))
        project = lambda point: (6371008.8 * math.radians(point[1]) * math.cos(lat0),
                                 6371008.8 * math.radians(point[0]))
        tree = cKDTree([project(coordinates[node]) for node in nodes])
        for record in parking.values():
            _, index = tree.query(project((record["lat"], record["lon"])))
            node = nodes[int(index)]
            snap = distance_m((record["lat"], record["lon"]), coordinates[node])
            accepted = snap <= geo["max_graph_snap_distance_m"]
            record.update(nearest_node=node, graph_snap_distance_m=snap, graph_snap_accepted=accepted,
                          graph_component=component_of[node],
                          runway_minimum_distance_m={name: float(values[node] + snap)
                              for name, values in distances.items() if accepted and node in values})
    else:
        for record in parking.values():
            record.update(graph_snap_accepted=False, runway_minimum_distance_m={})
    return {"parking": parking, "ambiguous_parking": ambiguous, "gates": gates,
            "ambiguous_gates": gate_ambiguous,
            "element_counts": dict(Counter(item.get("tags", {}).get("aeroway", "unknown") for item in elements)),
            "graph": {"nodes": len(nodes), "edges": graph.number_of_edges(),
                "connected_components": len(components), "largest_component_nodes": max(map(len, components), default=0),
                "runway_access_node_counts": {name: len(values) for name, values in sorted(accesses.items())}},
            "position_method": "Node coordinates or mean of way geometry; duplicate representatives must be within 30m. Means are approximate parking locations, not surveyed stand coordinates.",
            "route_interpretation": geo["route_proxy"]}


def coverage(frame, mapped):
    stands = frame.STAND_mvt.map(clean_ref)
    runways = frame.RUNWAY_mvt.map(runway_ref)
    parking = mapped["parking"]
    exact = stands.isin(parking)
    distances = []
    snapped = 0
    runway_matched = 0
    for stand, runway in zip(stands, runways):
        record = parking.get(stand)
        if record is None:
            continue
        snapped += bool(record["graph_snap_accepted"])
        runway_matched += runway in mapped["graph"]["runway_access_node_counts"]
        if runway in record["runway_minimum_distance_m"]:
            distances.append(record["runway_minimum_distance_m"][runway])
    n = len(frame)
    return {"rows": n, "distinct_stands": int(stands.nunique()),
        "exact_unambiguous_parking_rows": int(exact.sum()),
        "exact_unambiguous_parking_fraction": float(exact.mean()) if n else None,
        "ambiguous_parking_rows": int(stands.isin(mapped["ambiguous_parking"]).sum()),
        "passenger_gate_ref_match_rows_reported_separately": int(stands.isin(mapped["gates"]).sum()),
        "parking_graph_snap_rows": snapped, "parking_and_runway_ref_rows": runway_matched,
        "parking_to_recorded_runway_connected_rows": len(distances),
        "parking_to_recorded_runway_connected_fraction": len(distances) / n if n else None,
        "minimum_route_distance_quantiles_m": dict(zip(["min", "p10", "median", "p90", "max"],
            map(float, np.quantile(distances, [0, .1, .5, .9, 1])))) if distances else {},
        "unmatched_stand_row_counts": {str(k): int(v) for k, v in stands.loc[~exact].value_counts().items()}}


def load_snapshot(root, airport, stamp):
    label = "current" if stamp == "current" else stamp[:10].replace("-", "")
    name = f"{airport}_{label}.json"
    receipt = read_json(root / "docs/source_geometry/sources" / (name + ".receipt.json"))
    path = root / receipt["path"]
    if sha256_file(path) != receipt["sha256"] or receipt["requested_snapshot"] != stamp:
        raise ValueError("Geometry source or requested snapshot changed.")
    if receipt["declaration_sha256"] != sha256_file(root / CONFIG):
        raise ValueError("Geometry source belongs to another declaration.")
    raw = read_json(path)
    if stamp != "current":
        future = [item for item in raw["elements"] if item.get("timestamp", "") > stamp]
        if future:
            raise ValueError("Historical query contains later-edited OSM elements.")
    return raw["elements"], receipt


def main(root, *, complete):
    output = root / "docs/source_geometry" / ("geometry_coverage.json" if complete else "current_geometry_coverage.json")
    if output.exists():
        raise ValueError("Preserve the original geometry coverage report.")
    config = checked_config(root)
    with ResourceMonitor() as monitor:
        populations, population_receipt = read_geometry_covariates(root, config)
        populations["ranking"] = pd.concat([populations["2026-01"], populations["2026-07"]], ignore_index=True)
        airports, source_receipts, locations = {}, {}, {}
        for airport in config["airports"]:
            elements, receipt = load_snapshot(root, airport, "current")
            mapped = map_snapshot(elements, config)
            current = {name: coverage(frame.loc[frame.ADEP_mvt.eq(airport)], mapped) for name, frame in populations.items()}
            eligible = current["ranking"]["exact_unambiguous_parking_fraction"] >= config["geometry"]["historical_fetch_minimum_current_exact_parking_row_coverage"]
            airports[airport] = {"current": current, "current_graph": mapped["graph"],
                "current_element_counts": mapped["element_counts"],
                "current_valid_parking_refs": len(mapped["parking"]), "current_ambiguous_parking_refs": len(mapped["ambiguous_parking"]),
                "eligible_for_declared_historical_fetch": eligible, "historical": {}}
            source_receipts[f"{airport}/current"] = receipt
            locations[f"{airport}/current"] = mapped
            if complete and eligible:
                historical_maps = []
                for stamp in config["geometry"]["requested_snapshots"][1:]:
                    elements, receipt = load_snapshot(root, airport, stamp)
                    old = map_snapshot(elements, config)
                    month = stamp[:7]
                    frame = populations[month].loc[populations[month].ADEP_mvt.eq(airport)]
                    airports[airport]["historical"][stamp] = {"matching_month": coverage(frame, old),
                        "graph": old["graph"], "valid_parking_refs": len(old["parking"]),
                        "ambiguous_parking_refs": len(old["ambiguous_parking"])}
                    source_receipts[f"{airport}/{stamp}"] = receipt
                    locations[f"{airport}/{stamp}"] = old
                    historical_maps.append(old)
                shared = set.intersection(*[set(item["parking"]) for item in historical_maps])
                stable, moving = [], []
                for ref in sorted(shared):
                    points = [(item["parking"][ref]["lat"], item["parking"][ref]["lon"]) for item in historical_maps]
                    span = max(distance_m(a, b) for a in points for b in points)
                    (stable if span <= config["geometry"]["max_duplicate_position_span_m"] else moving).append(ref)
                rank = populations["ranking"].loc[populations["ranking"].ADEP_mvt.eq(airport)]
                count = int(rank.STAND_mvt.map(clean_ref).isin(stable).sum())
                airports[airport]["historical_position_stability"] = {
                    "refs_present_unambiguously_at_all_four_period_starts": len(shared),
                    "refs_within_30m_across_all_four_snapshots": len(stable),
                    "stable_refs": stable, "moved_more_than_30m_refs": moving,
                    "ranking_rows_with_stable_parking_ref": count, "ranking_fraction_with_stable_parking_ref": count / len(rank),
                    "interpretation": "Recorded OSM state at period starts; edit dates do not prove the operational layout or stability throughout a month."}
    details_path = output.with_name(output.stem + "_locations.json")
    if details_path.exists():
        raise ValueError("Preserve the original location detail report.")
    write_json(details_path, locations)
    report = {"created_at_utc": utc_now(), "complete_historical_check": complete,
        "declaration_sha256": sha256_file(root / CONFIG), "population": population_receipt,
        "airports": airports, "source_receipts": source_receipts, "resources": monitor.stats,
        "location_details_sha256": sha256_file(details_path),
        "code_sha256": {path.relative_to(root).as_posix(): sha256_file(path) for path in [
            Path(__file__), Path(__file__).with_name("common.py"), root / "tools/audit_airport_geometry.py"]},
        "audit_used": False, "labels_read": False, "models_fitted": 0, "predictions_changed": False,
        "license": "Map data copyright OpenStreetMap contributors, ODbL-1.0; source/license snapshots retained.",
        "interpretation": "Exact parking-reference and conservative graph coverage only. Passenger gate matches are separate. No inferred aliases, directed operational routing, or model-gain claim."}
    json.dumps(report, allow_nan=False)
    write_json(output, report)
    print(json.dumps({"complete_historical_check": complete, "airports": {name: {
        "ranking_current": item["current"]["ranking"], "historical": item["historical"],
        "stability": {k: v for k, v in item.get("historical_position_stability", {}).items() if not isinstance(v, list)}}
        for name, item in airports.items()}}, indent=2))
