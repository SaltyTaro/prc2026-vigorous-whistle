"""Guard protected periods and preserve signed/extreme source observations."""
import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, TIME, PHASE, TARGET
from prc2026_source_geometry.common import COVARIATES, validate_fitting_frame
from prc2026_source_geometry.source_audit import finite_summary, near_nonzero_multiple
from prc2026_source_geometry.geometry_audit import (
    clean_ref, coverage, graph_from_elements, map_snapshot, position_index, runway_ref,
)
from prc2026_source_geometry.geometry_review import resolved_positions, reviewed_map
from prc2026_source_geometry.endpoint_review import add_documented_endpoints, explicit_controls


def frame():
    return pd.DataFrame({ID: [1., 2., 3.], PHASE: ["DEP"] * 3,
        TIME: pd.to_datetime(["2025-02-01", "2025-08-01", "2025-09-30"], utc=True),
        TARGET: [-12, 900, 131167]})


def test_fitting_check_keeps_negative_and_multiday_labels():
    data = frame()
    before = data.copy(deep=True)
    validate_fitting_frame(data, {1., 2., 3.}, ["2025-02", "2025-08", "2025-09"])
    pd.testing.assert_frame_equal(data, before)
    summary = finite_summary(data[TARGET])
    assert summary["rows"] == 3
    assert summary["quantiles_seconds"]["min"] == -12
    assert summary["quantiles_seconds"]["max"] == 131167


@pytest.mark.parametrize("date", ["2025-01-01", "2025-07-01", "2025-10-01", "2025-11-01", "2025-12-01", "2026-01-01"])
def test_fitting_check_rejects_actual_utc_month_outside_common_fit(date):
    data = frame()
    data.loc[0, TIME] = pd.Timestamp(date, tz="UTC")
    with pytest.raises(ValueError, match="actual-UTC fitting"):
        validate_fitting_frame(data, {1., 2., 3.}, ["2025-02", "2025-08", "2025-09"])


@pytest.mark.parametrize("ids", [[1., 2., 2.], [1., 2., 4.], [1., 2., np.nan]])
def test_fitting_check_rejects_changed_or_duplicate_ids(ids):
    data = frame()
    data[ID] = ids
    with pytest.raises(ValueError, match="IDs"):
        validate_fitting_frame(data, {1., 2., 3.}, ["2025-02", "2025-08", "2025-09"])


def test_geometry_projection_has_no_departure_labels_or_block_times():
    assert TARGET not in COVARIATES
    assert "BLOCK_TIME_UTC_mvt" not in COVARIATES


def test_calendar_multiple_diagnostic_excludes_zero_and_retains_both_signs():
    values = [-86520., -86400., -86279., 0., 60., 86400., 86520., 172800., np.nan]
    assert near_nonzero_multiple(values, 86400, 120).tolist() == [True, True, False, False, False, True, True, True, False]


def node(number, ref, lat=49., lon=2., kind="parking_position"):
    return {"type": "node", "id": number, "lat": lat, "lon": lon,
            "tags": {"aeroway": kind, "ref": ref}}


def way(number, ids, points, kind="taxiway", ref=None):
    return {"type": "way", "id": number, "nodes": ids,
            "geometry": [{"lat": lat, "lon": lon} for lat, lon in points],
            "tags": {"aeroway": kind, **({"ref": ref} if ref else {})}}


def test_reference_cleaning_does_not_infer_stand_aliases():
    assert clean_ref(" d04 ") == "D04"
    assert clean_ref("004") != clean_ref("4")
    assert clean_ref("D04") != clean_ref("D4")
    assert clean_ref(None) is None
    assert clean_ref(" ") is None


def test_same_named_passenger_gate_is_not_a_parking_match():
    parking, ambiguous = position_index([node(1, "D04", kind="gate")], "parking_position", 30)
    assert parking == {} and ambiguous == {}


def test_distant_duplicate_stand_references_are_ambiguous():
    points = [node(1, "D04"), node(2, "D04", lat=49.005), node(3, "D05")]
    parking, ambiguous = position_index(points, "parking_position", 30)
    assert set(parking) == {"D05"}
    assert set(ambiguous) == {"D04"}
    assert ambiguous["D04"]["span_m"] > 500


def test_colocated_duplicate_objects_keep_one_exact_reference():
    points = [node(1, "D04"), node(2, "D04", lat=49.0001)]
    parking, ambiguous = position_index(points, "parking_position", 30)
    assert set(parking) == {"D04"} and not ambiguous
    assert len(parking["D04"]["sources"]) == 2


def test_runway_traversal_is_excluded_from_taxiway_graph():
    items = [way(1, [1, 2], [(49., 2.), (49., 2.001)]),
             way(2, [3, 4], [(49., 2.003), (49., 2.004)]),
             way(3, [2, 3], [(49., 2.001), (49., 2.003)], "runway", "09/27")]
    graph, _, access = graph_from_elements(items)
    assert graph.number_of_edges() == 2
    assert not graph.has_edge(2, 3)
    assert access == {"09": {2, 3}, "27": {2, 3}}


def test_crossing_coordinates_without_shared_osm_nodes_do_not_connect():
    items = [way(1, [1, 2], [(49., 2.), (49., 2.001)]),
             way(2, [3, 4], [(49., 2.001), (49., 2.002)])]
    graph, _, _ = graph_from_elements(items)
    assert not __import__("networkx").has_path(graph, 1, 4)


def test_runway_direction_names_normalize_only_valid_designators():
    assert runway_ref("8l") == "08L"
    assert runway_ref("NA") is None
    assert runway_ref("37") is None
    assert runway_ref("RWY08L") is None


def test_coverage_counts_real_parking_and_requires_connected_recorded_runway():
    items = [node(10, "D04"), node(11, "D05", kind="gate"),
             way(1, [1, 2], [(49., 2.), (49., 2.001)]),
             way(2, [2, 3], [(49., 2.001), (49., 2.004)], "runway", "09/27")]
    config = {"geometry": {"max_duplicate_position_span_m": 30, "max_graph_snap_distance_m": 75,
                            "route_proxy": "Undirected geometric lower bound."}}
    mapped = map_snapshot(items, config)
    rows = pd.DataFrame({"STAND_mvt": ["D04", "D04", "D05", "D4"],
                         "RUNWAY_mvt": ["09", "18", "09", "09"]})
    result = coverage(rows, mapped)
    assert result["rows"] == 4
    assert result["exact_unambiguous_parking_rows"] == 2
    assert result["passenger_gate_ref_match_rows_reported_separately"] == 1
    assert result["parking_to_recorded_runway_connected_rows"] == 1


def test_explicit_node_membership_resolves_long_same_ref_parking_line():
    items = [node(10, "D04"), way(1, [1, 10], [(49., 2.004), (49., 2.)], "parking_position", "D04")]
    before, before_ambiguous = position_index(items, "parking_position", 30)
    assert not before and "D04" in before_ambiguous
    parking, ambiguous = resolved_positions(items, 30)
    assert not ambiguous and parking["D04"]["lat"] == 49. and parking["D04"]["lon"] == 2.
    assert parking["D04"]["linked_same_ref_ways"][0]["same_ref_parking_node_members"] == [10]


def test_same_coordinates_without_node_identity_do_not_resolve_way_conflict():
    items = [node(10, "D04"), way(1, [1, 99], [(49., 2.004), (49., 2.)], "parking_position", "D04")]
    parking, ambiguous = resolved_positions(items, 30)
    assert "D04" in ambiguous and "D04" not in parking


def test_linked_ways_do_not_hide_conflicting_named_parking_nodes():
    items = [node(10, "D04"), node(11, "D04", lat=49.01),
             way(1, [1, 10], [(49., 2.004), (49., 2.)], "parking_position", "D04")]
    parking, ambiguous = resolved_positions(items, 30)
    assert "D04" in ambiguous and "D04" not in parking


def test_linked_parking_lead_in_connects_stand_without_large_geometric_snap():
    items = [node(10, "D04", lon=2.004),
             way(1, [1, 2], [(49., 2.), (49., 2.001)]),
             way(2, [1, 3], [(49., 2.), (49.002, 2.)], "runway", "09/27"),
             way(3, [2, 10], [(49., 2.001), (49., 2.004)], "parking_position", "D04")]
    config = {"geometry": {"max_duplicate_position_span_m": 30, "max_graph_snap_distance_m": 75,
                            "route_proxy": "Undirected geometric lower bound."}}
    result = reviewed_map(items, config)
    assert result["parking"]["D04"]["graph_snap_distance_m"] == 0
    assert result["parking"]["D04"]["runway_minimum_distance_m"]["09"] > 200
    assert result["parking_lead_in_way_ids_added_to_graph"] == [3]


@pytest.mark.parametrize("area", [True, False])
def test_parking_area_or_closed_outline_cannot_become_a_taxi_connector(area):
    items = [node(10, "D04", lon=2.004), way(1, [1, 2], [(49., 2.), (49., 2.001)])]
    extra = way(3, [2, 10] if area else [2, 10, 2],
                [(49., 2.001), (49., 2.004)] if area else [(49., 2.001), (49., 2.004), (49., 2.001)],
                "parking_position", "D04")
    if area:
        extra["tags"]["area"] = "yes"
    items.append(extra)
    config = {"geometry": {"max_duplicate_position_span_m": 30, "max_graph_snap_distance_m": 75,
                            "route_proxy": "Undirected geometric lower bound."}}
    result = reviewed_map(items, config)
    assert result["parking_lead_in_way_ids_added_to_graph"] == []


def test_endpoint_control_requires_the_explicit_nose_position_at_the_final_node():
    parked = node(10, "D04", lon=2.004)
    correct = way(1, [1, 10], [(49., 2.), (49., 2.004)], "parking_position", "D04")
    reversed_way = way(2, [10, 1], [(49., 2.004), (49., 2.)], "parking_position", "D04")
    records = explicit_controls([parked, correct, reversed_way], 1.)
    assert [item["last_node_convention_passes"] for item in records] == [True, False]
    assert records[0]["maximum_control_distance_to_last_node_m"] == 0
    assert records[1]["maximum_control_distance_to_last_node_m"] > 200


def test_endpoint_control_does_not_accept_coordinate_coincidence_or_ref_aliases():
    parked = node(10, "D04", lon=2.004)
    coordinate_only = way(1, [1, 99], [(49., 2.), (49., 2.004)], "parking_position", "D04")
    different_ref = way(2, [1, 10], [(49., 2.), (49., 2.004)], "parking_position", "D4")
    assert explicit_controls([parked, coordinate_only, different_ref], 1.) == []


def test_documented_endpoint_retains_original_source_and_flags_derived_location():
    import copy
    items = [way(1, [1, 10], [(49., 2.), (49., 2.004)], "parking_position", "D04")]
    before = copy.deepcopy(items)
    expanded, inferred, conflicts = add_documented_endpoints(items)
    assert items == before and len(expanded) == 2
    assert inferred == {"D04": [1]} and conflicts == []
    assert expanded[-1]["id"] == 10 and expanded[-1]["lon"] == 2.004
    assert expanded[-1]["derived_from_parking_way"] is True


def test_explicit_parking_node_prevents_way_endpoint_override():
    items = [node(20, "D04", lat=49.01),
             way(1, [1, 10], [(49., 2.), (49., 2.004)], "parking_position", "D04")]
    expanded, inferred, conflicts = add_documented_endpoints(items)
    assert expanded == items and inferred == {} and conflicts == []


def test_different_ref_explicit_endpoint_is_recorded_as_a_conflict():
    items = [node(10, "D05", lon=2.004),
             way(1, [1, 10], [(49., 2.), (49., 2.004)], "parking_position", "D04")]
    expanded, inferred, conflicts = add_documented_endpoints(items)
    assert inferred == {} and len(conflicts) == 1
    assert conflicts[0]["existing_parking_ref"] == "D05"
    assert all(not item.get("derived_from_parking_way") for item in expanded)
    assert expanded == [items[0]]


def test_two_stand_refs_cannot_be_inferred_from_the_same_endpoint():
    items = [way(1, [1, 10], [(49., 2.), (49., 2.004)], "parking_position", "D04"),
             way(2, [2, 10], [(49.001, 2.), (49., 2.004)], "parking_position", "D05")]
    expanded, inferred, conflicts = add_documented_endpoints(items)
    assert expanded == [] and inferred == {}
    assert {item["ref"] for item in conflicts} == {"D04", "D05"}


@pytest.mark.parametrize("area", [True, False])
def test_closed_or_area_parking_geometry_cannot_supply_a_documented_endpoint(area):
    item = way(1, [1, 10] if area else [1, 10, 1],
               [(49., 2.), (49., 2.004)] if area else [(49., 2.), (49., 2.004), (49., 2.)],
               "parking_position", "D04")
    if area:
        item["tags"]["area"] = "yes"
    expanded, inferred, conflicts = add_documented_endpoints([item])
    assert expanded == [item] and inferred == {} and conflicts == []
