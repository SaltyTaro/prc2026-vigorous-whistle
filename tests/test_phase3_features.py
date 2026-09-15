import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from prc2026.features import (BLOCK, FEATURE_NAMES, ID, PHASE, TARGET, TIME,
                              departure_features, movement_context, ranking_view)
from prc2026_phase3.bundle import load_partition, validate_feature_config
from prc2026_phase3.features import (ALL_FEATURE_NAMES, ARRIVAL_NUMERIC, INPUT_COLUMNS,
                                    TRAFFIC_NUMERIC, feature_contract, past_history,
                                    row_features, traffic_features)


def extended(movements):
    raw = movements.copy()
    raw["ADES_FILED_flt"] = raw["ADES_mvt"]
    raw["FLIGHT_RULE_flt"] = raw["FLIGHT_RULE_mvt"]
    raw["FLIGHT_mvt"] = ["BA123", "TK456", "LH789", "THY99"]
    raw["RUNWAY_mvt"] = ["09R", "27L", "07C", "36"]
    raw["STAND_mvt"] = ["218L", "A14", "A1", "E1"]
    return raw


def test_departure_labels_and_excluded_clocks_cannot_change_new_features(movements):
    raw = extended(movements)
    changed = raw.copy()
    dep = changed[PHASE].eq("DEP")
    changed.loc[dep, TARGET] = [888888, -8888, 0]
    changed.loc[dep, BLOCK] += pd.Timedelta(days=300)
    for name in ["AOBT_3_flt", "LOBT_flt", "ARVT_3_flt"]:
        changed[name] += pd.Timedelta(days=500)
    original = row_features(raw)
    assert_frame_equal(original, row_features(changed))
    assert_frame_equal(original[[ID, *FEATURE_NAMES]], departure_features(ranking_view(raw)))
    q = raw.loc[raw[PHASE].eq("DEP")]
    assert_frame_equal(traffic_features(movement_context(ranking_view(raw)), q),
                       traffic_features(movement_context(ranking_view(changed)), q))
    assert not {TARGET, BLOCK, "AOBT_3_flt", "LOBT_flt", "ARVT_3_flt", "FLIGHT_ID_mvt"} & set(INPUT_COLUMNS)
    assert len(ALL_FEATURE_NAMES) == len(set(ALL_FEATURE_NAMES))


def test_airport_codes_and_local_clock_keep_meaning(movements):
    features = row_features(extended(movements)).set_index(ID)
    assert features.loc[101, "airport_stand"] == "EGLL|218L"
    assert features.loc[101, "airport_stand_area"] == "EGLL|#2"
    assert features.loc[102, "airport_stand_area"] == "EGLL|A"
    assert features.loc[101, "stand_number"] == 218
    assert features.loc[101, "runway_number"] == 9
    assert features.loc[101, "runway_heading_sin"] == pytest.approx(1)
    assert features.loc[101, "utc_offset_hours"] == 0
    assert features.loc[102, "utc_offset_hours"] == 1
    assert features.loc[104, "utc_offset_hours"] == 3
    assert features.loc[101, "local_minute_of_day"] == 30
    assert features.loc[102, "local_minute_of_day"] == 150
    assert features.loc[104, "local_minute_of_day"] == 90


def test_history_exact_boundaries_ties_and_airport_isolation():
    t = pd.Timestamp("2025-01-01T12:00:00Z")
    events = pd.DataFrame({"airport": ["A"]*5 + ["B"],
                           TIME: [t+pd.Timedelta(seconds=s) for s in [-301, -300, -1, 0, 1, -10]],
                           "value": [999., 10., 30., 999., 999., 999.]})
    q = pd.DataFrame({"airport": ["A", "B", "C"], TIME: [t]*3})
    result = past_history(events, q, ["airport"], "example", (5,), "value")
    assert result.loc[0, "example_count_5m"] == 2
    assert result.loc[0, "example_last_s"] == 1
    assert result.loc[0, "example_mean_5m"] == 20
    assert result.loc[0, "example_std_5m"] == 10
    assert result.loc[1, "example_count_5m"] == 1
    assert result.loc[2, "example_count_5m"] == 0
    assert np.isnan(result.loc[2, "example_mean_5m"])
    shuffled = past_history(events.sample(frac=1, random_state=4), q, ["airport"], "example", (5,), "value")
    assert_frame_equal(result, shuffled)


def test_history_does_not_depend_on_future_context():
    t = pd.Timestamp("2025-01-01T12:00:00Z")
    past = pd.DataFrame({"airport": ["A"], TIME: [t-pd.Timedelta(minutes=1)]})
    future = pd.DataFrame({"airport": ["A", "A"], TIME: [t, t+pd.Timedelta(seconds=1)]})
    q = pd.DataFrame({"airport": ["A"], TIME: [t]})
    assert_frame_equal(past_history(past, q, ["airport"], "event", (5,)),
                       past_history(pd.concat([past, future]), q, ["airport"], "event", (5,)))


def test_unobserved_gaps_and_missing_values_are_explicit():
    t = pd.Timestamp("2025-01-01T12:00:00Z")
    events = pd.DataFrame({"airport": ["A", "B"], TIME: [t-pd.Timedelta(hours=7), t-pd.Timedelta(seconds=1)],
                           "value": [100., np.nan]})
    q = pd.DataFrame({"airport": ["A", "B"], TIME: [t, t]})
    out = past_history(events, q, ["airport"], "event", (60,), "value")
    assert np.isnan(out.loc[0, "event_last_s"])
    assert out.loc[0, "event_count_60m"] == 0
    assert out.loc[1, "event_count_60m"] == 1
    assert np.isnan(out.loc[1, "event_mean_60m"])


def test_arrival_duration_is_unavailable_until_in_block(movements):
    raw = extended(movements)
    t = pd.Timestamp("2025-03-30T01:50:00Z")
    raw.loc[0, TIME] = t
    raw.loc[0, "ADEP_mvt"] = "EDDF"
    raw.loc[2, "RUNWAY_mvt"] = raw.loc[0, "RUNWAY_mvt"]
    raw.loc[2, "STAND_mvt"] = raw.loc[0, "STAND_mvt"]
    # The arrival lands at 01:45 and reaches its stand at 01:55.
    raw.loc[2, BLOCK] = t + pd.Timedelta(minutes=5)
    raw.loc[2, TARGET] = 600
    ctx = movement_context(ranking_view(raw))
    before = traffic_features(ctx, raw.iloc[[0]])
    assert before.loc[0, "airport_arr_count_15m"] == 1
    assert before.loc[0, "completed_arrival_taxi_count_30m"] == 0
    assert np.isnan(before.loc[0, "completed_arrival_taxi_mean_30m"])
    query = raw.iloc[[0]].copy()
    query[TIME] = t + pd.Timedelta(minutes=6)
    after = traffic_features(ctx, query)
    assert after.loc[0, "completed_arrival_taxi_count_30m"] == 1
    assert after.loc[0, "completed_arrival_taxi_mean_30m"] == 600
    assert list(after) == [ID, *TRAFFIC_NUMERIC, *ARRIVAL_NUMERIC]


def test_contaminated_departure_context_is_rejected(movements):
    raw = extended(movements)
    ctx = movement_context(ranking_view(raw))
    ctx.loc[0, "arrival_taxi_in_s"] = 600
    with pytest.raises(ValueError, match="masked"):
        traffic_features(ctx, raw.iloc[[0]])


def test_feature_config_cannot_materialize_audit(protocol):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/features_v1.json").read_text())
    original = {"bundle_id": config["parent_bundle_id"], "descriptor": {"protocol": protocol}}
    validate_feature_config(config, original)
    config["training_months"].append("2025-12")
    with pytest.raises(ValueError, match="reserved audit"):
        validate_feature_config(config, original)


def test_feature_loader_refuses_audit_before_reading_labels(tmp_path):
    catalog = pd.DataFrame({ID: [1.], TIME: pd.to_datetime(["2025-12-01T00:00:00Z"]),
                            "source_file": ["unread.parquet"]})
    assignment = pd.DataFrame({ID: [1.], "partition": ["score"]})
    with pytest.raises(ValueError, match="reserved audit"):
        load_partition(tmp_path, tmp_path, catalog, assignment, "score", feature_contract([])[0])
