"""Protect time boundaries, route identities, peer exclusion and v7 fallback."""

import math
import numpy as np
import pandas as pd
import pytest
import networkx as nx

from prc2026.features import ID, TIME, PHASE, TARGET, BLOCK
from prc2026_timeline.encoding import contract as old_contract, fit_encoder as old_encoder, transform as old_transform
from prc2026_rome_geometry.features import (
    INPUT_COLUMNS, STATIC_NAMES, QUEUE_NAMES, EXTRA_NAMES, checked_context, build_map,
    deterministic_paths, make_features, queue_features,
)
from prc2026_rome_geometry.encoding import contract, fit_encoder, transform
from prc2026_rome_geometry.inference import compose


def config():
    return {"airport":"LIRF", "snapshot_assignment":{"2025-02":"2025-01-01T00:00:00Z"},
            "coordinate_origin_lat_lon":[41.81,12.25]}


def raw(rows=None):
    rows = rows or [(1., 101., "12:20:00", "12:00:00", "A1")]
    frame = pd.DataFrame(rows, columns=[ID,"FLIGHT_ID_mvt",TIME,"AOBT_3_flt","STAND_mvt"])
    for name in [TIME,"AOBT_3_flt"]:
        frame[name] = pd.to_datetime("2025-02-01 " + frame[name], utc=True)
    frame["LOBT_flt"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    frame[PHASE], frame["ADEP_mvt"], frame["RUNWAY_mvt"] = "DEP", "LIRF", "09"
    return frame[INPUT_COLUMNS]


def node(number, ref, lat=41.81, lon=12.253):
    return {"type":"node", "id":number, "lat":lat, "lon":lon, "tags":{"aeroway":"parking_position","ref":ref}}


def way(number, ids, points, kind="taxiway", ref=None):
    return {"type":"way", "id":number, "nodes":ids, "geometry":[{"lat":a,"lon":b} for a,b in points],
            "tags":{"aeroway":kind, **({"ref":ref} if ref else {})}}


def toy_map():
    elements = [node(10,"A1"),
        way(1,[1,2],[(41.81,12.25),(41.81,12.251)]),
        way(2,[2,10],[(41.81,12.251),(41.81,12.253)],"parking_position","A1"),
        way(3,[1,3],[(41.81,12.25),(41.811,12.25)],"runway","09/27")]
    review = {"geometry":{"max_duplicate_position_span_m":30,"max_graph_snap_distance_m":75,
                          "route_proxy":"Undirected mapped minimum."}}
    return build_map(elements, review, config())


def queue_map():
    return {"route_index":{("A1","09"):0,("B1","09"):1},"shared_fraction":np.eye(2)}


def test_geometry_contract_excludes_targets_and_block_times():
    assert TARGET not in INPUT_COLUMNS and BLOCK not in INPUT_COLUMNS
    assert ID not in EXTRA_NAMES and "FLIGHT_ID_mvt" not in EXTRA_NAMES
    assert len(STATIC_NAMES) == 15 and len(QUEUE_NAMES) == 15


@pytest.mark.parametrize("label", [TARGET, BLOCK])
def test_shared_features_require_departure_label_masking(label):
    frame = raw()
    frame[label] = 99999
    with pytest.raises(ValueError, match="masked"):
        checked_context(frame, config())


@pytest.mark.parametrize("month", ["2025-12","2026-02"])
def test_actual_movement_month_protects_december_and_undeclared_periods(month):
    frame = raw()
    frame[TIME] = pd.to_datetime([month + "-01"], utc=True)
    with pytest.raises(ValueError, match="Reserved December"):
        checked_context(frame, config())


def test_snapshot_cannot_postdate_movement():
    declaration = config()
    declaration["snapshot_assignment"]["2025-02"] = "2025-03-01T00:00:00Z"
    with pytest.raises(ValueError, match="postdate"):
        checked_context(raw(), declaration)


def test_only_focal_airport_departures_enter_new_geometry():
    frame = raw()
    frame["ADEP_mvt"] = "LFPG"
    with pytest.raises(ValueError, match="LIRF departures"):
        checked_context(frame, config())


def test_duplicate_movement_ids_are_rejected():
    with pytest.raises(ValueError, match="Unique integral"):
        checked_context(pd.concat([raw(),raw()],ignore_index=True), config())


@pytest.mark.parametrize("offblock, expected_anchor", [("12:30:00","12:00:00"),("03:00:00","12:00:00"),("12:20:00","12:20:00")])
def test_signed_clock_intervals_remain_unchanged_with_declared_anchor_fallback(offblock, expected_anchor):
    frame = raw([(1.,101.,"12:20:00",offblock,"A1")])
    data = checked_context(frame,config())
    assert data.anchor.iloc[0] == pd.Timestamp("2025-02-01 " + expected_anchor, tz="UTC")
    assert data.AOBT_3_flt.iloc[0] == frame.AOBT_3_flt.iloc[0]


def test_lobt_is_used_only_when_aobt_is_missing():
    frame = raw()
    frame["LOBT_flt"] = pd.to_datetime(["2025-02-01 11:55:00"],utc=True)
    assert checked_context(frame,config()).anchor.iloc[0] == frame.AOBT_3_flt.iloc[0]
    frame["AOBT_3_flt"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    assert checked_context(frame,config()).anchor.iloc[0] == frame.LOBT_flt.iloc[0]


def test_shortest_path_ties_prefer_smallest_access_and_never_form_cycles():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(2,4,1.),(1,3,.5),(3,4,.5),(4,5,0.)])
    distances, parents = deterministic_paths(graph,[2,1])
    assert distances[4] == (1.,1) and distances[5] == (1.,1)
    for node_ in graph:
        seen=set()
        while node_ is not None:
            assert node_ not in seen
            seen.add(node_)
            node_=parents[node_]


def test_mapped_route_uses_parking_lead_in_but_not_runway_traversal():
    mapped=toy_map()
    route=mapped["routes"][("A1","09")]
    assert route["nodes"] == [10,2,1]
    assert route["features"]["geo_route_m"] > 200
    assert route["features"]["geo_route_junctions"] == 0
    assert route["features"]["geo_route_edges"] == 2
    assert (1,3) not in route["edges"]
    assert np.allclose(mapped["shared_fraction"],1.)


def test_exact_stand_refs_do_not_infer_zero_padding_aliases():
    source=raw([(1.,101.,"12:20:00","12:00:00","A01")])
    mapped=toy_map()
    output=make_features(source,[1.],{"2025-01-01T00:00:00Z":mapped},config())
    assert output.geo_stand_available.iloc[0] == 0
    assert output.geo_route_available.iloc[0] == 0
    assert output[QUEUE_NAMES].isna().all().all()


def test_queue_uses_prior_events_excludes_self_same_flight_and_unshared_routes():
    source=raw([
        (1.,101.,"12:20:00","12:00:00","A1"),
        (2.,102.,"12:10:00","11:59:00","A1"),
        (3.,103.,"12:10:00","12:00:00","A1"),
        (4.,104.,"12:10:00","12:01:00","A1"),
        (5.,101.,"12:10:00","11:58:00","A1"),
        (6.,106.,"11:50:00","11:40:00","A1"),
        (7.,107.,"12:10:00","11:59:00","B1"),
        (8.,108.,"12:00:00","11:50:00","A1"),
    ])
    data=checked_context(source,config())
    row=queue_features(data,data.iloc[[0]].reset_index(drop=True),queue_map()).iloc[0]
    assert row.geo_shared_push_count_15m == 2  # Includes the peer departing exactly at the anchor.
    assert row.geo_shared_push_count_60m == 3
    assert row.geo_shared_push_last_s == 60
    assert row.geo_shared_active_count == 2
    assert row.geo_shared_active_mean_age_s == 330
    assert row.geo_shared_service_count_15m == 1
    assert row.geo_shared_service_proxy_mean_s_15m == 600


def test_completed_service_retains_negative_and_very_large_nm_proxies():
    source=raw([(1.,101.,"12:20:00","12:00:00","A1"),
                (2.,102.,"11:59:00","12:10:00","A1"),
                (3.,103.,"11:58:00","01:00:00","A1")])
    data=checked_context(source,config())
    row=queue_features(data,data.iloc[[0]].reset_index(drop=True),queue_map()).iloc[0]
    assert row.geo_shared_service_count_15m == 2
    assert row.geo_shared_service_proxy_mean_s_15m == (-660+39480)/2
    assert row.geo_shared_active_count == 0


def test_push_lower_window_boundary_is_included():
    source=raw([(1.,101.,"12:20:00","12:00:00","A1"),
                (2.,102.,"12:10:00","11:45:00","A1"),
                (3.,103.,"12:10:00","11:44:59","A1")])
    data=checked_context(source,config())
    row=queue_features(data,data.iloc[[0]].reset_index(drop=True),queue_map()).iloc[0]
    assert row.geo_shared_push_count_15m == 1 and row.geo_shared_push_count_60m == 2


def test_geometry_build_ignores_query_order_for_feature_values():
    source=raw([(1.,101.,"12:20:00","12:00:00","A1"),(2.,102.,"12:10:00","11:59:00","A1")])
    maps={"2025-01-01T00:00:00Z":toy_map()}
    a=make_features(source,[1.,2.],maps,config()).set_index(ID).sort_index()
    b=make_features(source,[2.,1.],maps,config()).set_index(ID).sort_index()
    pd.testing.assert_frame_equal(a,b,check_exact=True)


def test_control_encoder_matches_frozen_transform_exactly():
    names,cats=contract("control")
    assert (names,cats)==old_contract({"features":"combined","target":"nm_residual"})
    data=pd.DataFrame({name:(["A",None] if name in cats else [1.,np.nan]) for name in names})
    data[ID]=[1.,2.]
    spec={"features":"combined","target":"nm_residual","kind":"lightgbm","route":"clock_available"}
    current,old=fit_encoder(data,"control"),old_encoder(data,spec)
    pd.testing.assert_frame_equal(transform(data,current),old_transform(data,old),check_exact=True)
    assert current["vocabularies"] == old["vocabularies"]


def test_composition_preserves_every_ineligible_value_and_signed_predictions():
    reference=np.array([-12.,500.,100000.])
    actual=compose(reference,np.array([2.,20.,8.]),np.array([10.,40.,18.]),np.array([False,True,False]),.5)
    assert np.array_equal(actual[[0,2]],reference[[0,2]])
    assert actual[1] == 505.


@pytest.mark.parametrize("weight", [0.,.25,.75,1.1])
def test_undeclared_candidate_weights_are_rejected(weight):
    with pytest.raises(ValueError,match="declared weight"):
        compose([1.],[2.],[3.],np.array([True]),weight)


def test_composition_rejects_shape_or_route_misalignment():
    with pytest.raises(ValueError,match="identical shapes"):
        compose([1.],[2.],[3.],np.array([1.]),1.)


def test_independent_queue_verifier_handles_pandas_read_only_boolean_views():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"tools/check_rome_geometry_features.py"
    spec=importlib.util.spec_from_file_location("rome_feature_verifier_test",path)
    verifier=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    data=checked_context(raw([(1.,101.,"12:20:00","12:00:00","A1"),
                              (2.,102.,"12:10:00","11:59:00","A1")]),config())
    mapped=queue_map()
    mapped["routes"]={("A1","09"):{},("B1","09"):{}}
    result=verifier.independent_queue(data,data.iloc[0],mapped)
    assert result["geo_shared_push_count_15m"] == 1
    assert result["geo_shared_active_count"] == 1
