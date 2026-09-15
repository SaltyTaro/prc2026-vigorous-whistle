"""Supported historical routes and strictly prior other-flight route context."""

from __future__ import annotations
import heapq
import math

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from prc2026.features import ID, TIME, PHASE, TARGET, BLOCK
from prc2026_source_geometry.geometry_audit import clean_ref, distance_m, graph_from_elements, runway_ref
from prc2026_source_geometry.geometry_review import reviewed_map
from prc2026_source_geometry.endpoint_review import add_documented_endpoints

POLICY = "historical-lirf-routes-and-prior-shared-route-context-v1"
INPUT_COLUMNS = [ID, "FLIGHT_ID_mvt", PHASE, TIME, "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "AOBT_3_flt", "LOBT_flt"]
STATIC_NAMES = ["geo_stand_available", "geo_route_available", "geo_endpoint_inferred", "geo_stand_east_m",
    "geo_stand_north_m", "geo_snap_m", "geo_route_m", "geo_direct_m", "geo_detour_ratio", "geo_route_edges",
    "geo_route_junctions", "geo_turn_radians", "geo_sharp_turns", "geo_access_east_m", "geo_access_north_m"]
QUEUE_NAMES = [f"geo_shared_push_{measure}_{window}m" for window in [15, 60] for measure in ["count", "equivalent", "max_fraction"]]
QUEUE_NAMES += ["geo_shared_push_last_s", "geo_shared_active_count", "geo_shared_active_equivalent",
                "geo_shared_active_max_age_s", "geo_shared_active_mean_age_s"]
QUEUE_NAMES += [f"geo_shared_service_{measure}_{window}m" for window in [15, 60] for measure in ["count", "proxy_mean_s"]]
EXTRA_NAMES = STATIC_NAMES + QUEUE_NAMES


def checked_context(raw, config):
    if TARGET in raw or BLOCK in raw:
        raise ValueError("Departure labels and block times must be masked before geometry generation.")
    data = raw[INPUT_COLUMNS].reset_index(drop=True).copy()
    ids = data[ID].to_numpy(dtype=float)
    if not np.isfinite(ids).all() or np.any(ids % 1) or data[ID].duplicated().any():
        raise ValueError("Unique integral movement IDs are required.")
    if not data[PHASE].eq("DEP").all() or not data.ADEP_mvt.eq(config["airport"]).all():
        raise ValueError("This feature extension is restricted to LIRF departures.")
    for name in [TIME, "AOBT_3_flt", "LOBT_flt"]:
        if not isinstance(data[name].dtype, pd.DatetimeTZDtype) or str(data[name].dt.tz) != "UTC":
            raise ValueError("Movement and supplied NM clocks require UTC types.")
    months = data[TIME].dt.strftime("%Y-%m")
    if data[TIME].isna().any() or not months.isin(config["snapshot_assignment"]).all():
        raise ValueError("Reserved December or undeclared movement month.")
    data["snapshot"] = months.map(config["snapshot_assignment"])
    if (pd.to_datetime(data.snapshot, utc=True) > data[TIME]).any():
        raise ValueError("A map snapshot must not postdate the actual movement.")
    data["stand"] = data.STAND_mvt.map(clean_ref)
    data["runway"] = data.RUNWAY_mvt.map(runway_ref)
    data["offblock"] = data.AOBT_3_flt.combine_first(data.LOBT_flt)
    data["proxy_s"] = (data[TIME] - data.offblock).dt.total_seconds()
    data["anchor"] = data.offblock.where(data.proxy_s.between(0, 21600), data[TIME] - pd.Timedelta(seconds=1200))
    return data


def project(point, origin):
    lat, lon = point
    return (6371008.8 * math.radians(lon - origin[1]) * math.cos(math.radians(origin[0])),
            6371008.8 * math.radians(lat - origin[0]))


def deterministic_paths(graph, accesses):
    """Distance, access ID and node ID determine equal-cost traversal order."""
    best, parent, settled, queue = {}, {}, set(), []
    for node in sorted(accesses):
        best[node], parent[node] = (0., node), None
        heapq.heappush(queue, (0., node, node))
    while queue:
        dist, access, node = heapq.heappop(queue)
        if node in settled or best.get(node) != (dist, access):
            continue
        settled.add(node)
        for neighbor in sorted(graph[node]):
            if neighbor in settled:
                continue
            candidate = (dist + graph[node][neighbor]["weight"], access)
            old = best.get(neighbor, (math.inf, math.inf))
            if candidate < old or (candidate == old and (parent.get(neighbor) is None or node < parent[neighbor])):
                best[neighbor], parent[neighbor] = candidate, node
                heapq.heappush(queue, (*candidate, neighbor))
    return best, parent


def build_map(elements, review_config, feature_config):
    expanded, inferred, conflicts = add_documented_endpoints(elements)
    if conflicts:
        raise ValueError("Unexpected LIRF endpoint conflict; retain the frozen audit.")
    mapped = reviewed_map(expanded, review_config)
    parking = {ref: value for ref, value in mapped["parking"].items()
               if value["sources"] and all(item["method"] == "node" for item in value["sources"])}
    linked = set(mapped["parking_lead_in_way_ids_added_to_graph"])
    graph_elements = [{**item, "tags": {**item.get("tags", {}), "aeroway": "taxilane"}}
                      if item["type"] == "way" and item["id"] in linked else item for item in expanded]
    graph, coordinates, access_nodes = graph_from_elements(graph_elements)
    origin = feature_config["coordinate_origin_lat_lon"]
    paths = {runway: deterministic_paths(graph, nodes) for runway, nodes in sorted(access_nodes.items())}
    routes, points = {}, {}
    for ref, record in sorted(parking.items()):
        east, north = project((record["lat"], record["lon"]), origin)
        point = {name: np.nan for name in STATIC_NAMES}
        point.update(geo_stand_available=1., geo_route_available=0., geo_endpoint_inferred=float(ref in inferred),
                     geo_stand_east_m=east, geo_stand_north_m=north, geo_snap_m=record["graph_snap_distance_m"])
        points[ref] = point
        if not record["graph_snap_accepted"]:
            continue
        node = record["nearest_node"]
        for runway, (distances, predecessors) in paths.items():
            if node not in distances:
                continue
            chain, current = [], node
            while current is not None:
                if current in chain:
                    raise ValueError("A shortest path contains a cycle.")
                chain.append(current)
                current = predecessors[current]
            length = distances[node][0] + record["graph_snap_distance_m"]
            if abs(length - record["runway_minimum_distance_m"][runway]) > 1e-7:
                raise ValueError("The route distance differs from the audited shortest distance.")
            end = coordinates[chain[-1]]
            direct = distance_m((record["lat"], record["lon"]), end)
            planar = [project(coordinates[n], origin) for n in chain]
            directions = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(planar[:-1], planar[1:]) if a != b]
            turns = [abs(math.atan2(math.sin(b - a), math.cos(b - a))) for a, b in zip(directions[:-1], directions[1:])]
            ex, ny = project(end, origin)
            values = {**point, "geo_route_available":1., "geo_route_m":length, "geo_direct_m":direct,
                "geo_detour_ratio":length / max(1., direct), "geo_route_edges":float(max(0, len(chain) - 1)),
                "geo_route_junctions":float(sum(graph.degree[n] >= 3 for n in chain)),
                "geo_turn_radians":float(sum(turns)), "geo_sharp_turns":float(sum(t >= math.pi / 4 for t in turns)),
                "geo_access_east_m":ex, "geo_access_north_m":ny}
            edges = [tuple(sorted((a, b))) for a, b in zip(chain[:-1], chain[1:])]
            routes[(ref, runway)] = {"features":values, "nodes":chain, "edges":edges}
    keys = sorted(routes)
    edge_keys = sorted({edge for item in routes.values() for edge in item["edges"]})
    edge_index = {edge:i for i, edge in enumerate(edge_keys)}
    rows, columns = [], []
    for i, key in enumerate(keys):
        for edge in routes[key]["edges"]:
            rows.append(i)
            columns.append(edge_index[edge])
    incidence = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(keys), len(edge_keys)))
    weights = np.array([graph[a][b]["weight"] for a, b in edge_keys])
    shared = (incidence.multiply(weights) @ incidence.T).toarray()
    lengths = np.array([routes[key]["features"]["geo_route_m"] for key in keys])
    fraction = np.divide(shared, lengths[:, None], out=np.zeros_like(shared), where=lengths[:, None] > 0)
    if np.any(fraction < -1e-9) or np.any(fraction > 1 + 1e-9):
        raise ValueError("Impossible shared-route length fraction.")
    return {"parking":points, "routes":routes, "route_keys":keys,
            "route_index":{key:i for i, key in enumerate(keys)}, "shared_fraction":fraction,
            "graph_summary":mapped["graph"], "inferred_refs":sorted(inferred)}


def seconds(stamps):
    values = stamps.to_numpy(dtype="datetime64[ns]").view("int64").astype(float) / 1e9
    values[stamps.isna().to_numpy()] = np.nan
    return values


def peer_mask(indexes, query, ids, flights):
    keep = ids[indexes] != query[ID]
    if pd.notna(query["FLIGHT_ID_mvt"]):
        keep &= flights[indexes] != query["FLIGHT_ID_mvt"]
    return keep


def queue_features(context, queries, mapped):
    result = np.full((len(queries), len(QUEUE_NAMES)), np.nan, dtype=float)
    positions = {name:i for i, name in enumerate(QUEUE_NAMES)}
    route_indexes = np.array([mapped["route_index"].get((stand, runway), -1) for stand, runway in zip(context.stand, context.runway)], dtype=int)
    starts, ends = seconds(context.offblock), seconds(context[TIME])
    ids, flights = context[ID].to_numpy(), context.FLIGHT_ID_mvt.to_numpy()
    proxies = context.proxy_s.to_numpy(dtype=float)
    push = np.flatnonzero((route_indexes >= 0) & np.isfinite(starts))
    push = push[np.lexsort((ids[push], starts[push]))]
    takeoff = np.flatnonzero(route_indexes >= 0)
    takeoff = takeoff[np.lexsort((ids[takeoff], ends[takeoff]))]
    push_times, takeoff_times = starts[push], ends[takeoff]
    anchors = seconds(queries.anchor)
    for qpos, (_, query) in enumerate(queries.iterrows()):
        route = mapped["route_index"].get((query.stand, query.runway))
        if route is None:
            continue
        anchor = anchors[qpos]
        lo, hi = np.searchsorted(push_times, [anchor - 21600, anchor], side="left")
        candidates = push[lo:hi]
        candidates = candidates[peer_mask(candidates, query, ids, flights)]
        overlap = mapped["shared_fraction"][route, route_indexes[candidates]]
        eligible = overlap > 0
        candidates, overlap = candidates[eligible], overlap[eligible]
        ages = anchor - starts[candidates]
        for window in [15, 60]:
            selected = ages <= window * 60
            weights = overlap[selected]
            for measure, value in [("count", len(weights)), ("equivalent", weights.sum()),
                                   ("max_fraction", weights.max() if len(weights) else 0.)]:
                result[qpos, positions[f"geo_shared_push_{measure}_{window}m"]] = value
        result[qpos, positions["geo_shared_push_last_s"]] = ages.min() if len(ages) else np.nan
        active = (ends[candidates] >= anchor) & (proxies[candidates] > 0) & (proxies[candidates] <= 21600)
        result[qpos, positions["geo_shared_active_count"]] = active.sum()
        result[qpos, positions["geo_shared_active_equivalent"]] = overlap[active].sum()
        result[qpos, positions["geo_shared_active_max_age_s"]] = ages[active].max() if active.any() else np.nan
        result[qpos, positions["geo_shared_active_mean_age_s"]] = ages[active].mean() if active.any() else np.nan
        lo, hi = np.searchsorted(takeoff_times, [anchor - 3600, anchor], side="left")
        candidates = takeoff[lo:hi]
        candidates = candidates[peer_mask(candidates, query, ids, flights)]
        candidates = candidates[mapped["shared_fraction"][route, route_indexes[candidates]] > 0]
        ages = anchor - ends[candidates]
        for window in [15, 60]:
            selected = candidates[ages <= window * 60]
            finite = proxies[selected][np.isfinite(proxies[selected])]
            result[qpos, positions[f"geo_shared_service_count_{window}m"]] = len(selected)
            result[qpos, positions[f"geo_shared_service_proxy_mean_s_{window}m"]] = finite.mean() if len(finite) else np.nan
    return pd.DataFrame(result, columns=QUEUE_NAMES)


def make_features(raw_context, query_ids, maps, config):
    context = checked_context(raw_context, config)
    queries = pd.DataFrame({ID:query_ids}).merge(context, on=ID, how="left", validate="one_to_one", sort=False)
    if queries[TIME].isna().any():
        raise ValueError("Every focal feature ID must be a declared LIRF departure.")
    out = pd.DataFrame(np.nan, index=queries.index, columns=EXTRA_NAMES, dtype=float)
    out["geo_stand_available"] = 0.
    out["geo_route_available"] = 0.
    for stamp, rows in queries.groupby("snapshot", sort=True):
        mapped = maps[stamp]
        for index, row in rows.iterrows():
            route = mapped["routes"].get((row.stand, row.runway))
            values = route["features"] if route else mapped["parking"].get(row.stand)
            if values:
                out.loc[index, STATIC_NAMES] = [values[name] for name in STATIC_NAMES]
        out.loc[rows.index, QUEUE_NAMES] = queue_features(context, rows.reset_index(drop=True), mapped).to_numpy()
    result = pd.concat([queries[[ID]], out.astype("float32")], axis=1)
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("Infinite geometry feature.")
    return result[[ID, *EXTRA_NAMES]]
