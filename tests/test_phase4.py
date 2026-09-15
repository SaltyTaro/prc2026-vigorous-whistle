from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc2026.features import BLOCK, ID, TARGET, TIME
from prc2026_phase4.bundle import scientific_parent
from prc2026_phase4.encoding import aligned_target, fit_encoder, transform
from prc2026_phase4.features import (CATEGORICAL_COLUMNS, FEATURE_NAMES, INPUT_COLUMNS,
                                     prediction_offset, quality_features, route_mask)
from prc2026_phase4.training import fit_node, pipeline_prediction, validate_config


def configuration():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/phase4_v1.json").read_text())


def raw_example(movements):
    raw = movements.copy()
    raw["FLIGHT_mvt"] = ["ITYTY630", "AMX071", "ARR001", None]
    raw.loc[0, "ADEP_mvt"] = "LIRF"
    raw.loc[0, ["ADEP_flt", "AIRCRAFT_TYPE_mvt"]] = None
    return raw


def sample_features(n=400):
    rng = np.random.default_rng(18)
    values = {ID: np.arange(n, dtype=float)}
    for name in FEATURE_NAMES:
        values[name] = rng.choice(["A", "B"], n) if name in CATEGORICAL_COLUMNS else rng.normal(size=n).astype("float32")
    frame = pd.DataFrame(values)
    frame["airport"] = np.where(np.arange(n) % 2 == 0, "LIRF", "EGLL")
    frame["lirf_missing_record"] = frame["airport"].eq("LIRF").astype("float32")
    frame["schedule_to_movement_s"] = np.arange(n, dtype="float32")*500
    frame["conditional_schedule_offset_s"] = frame["lirf_missing_record"]*frame["schedule_to_movement_s"]
    return frame


def test_quality_features_never_consume_targets_or_actual_clocks(movements):
    raw = raw_example(movements)
    expected = quality_features(raw)
    forbidden = [TARGET, BLOCK, "AOBT_3_flt", "LOBT_flt", "ARVT_3_flt", "FLIGHT_ID_mvt"]
    assert not set(forbidden) & set(INPUT_COLUMNS)
    for name in forbidden:
        raw[name] = "unavailable or adversarial target"
    pd.testing.assert_frame_equal(expected, quality_features(raw), check_exact=True)
    assert list(expected[ID]) == [101.0, 102.0, 104.0]


def test_schedule_gate_requires_all_input_conditions_and_finite_clock(movements):
    raw = raw_example(movements)
    extra = quality_features(raw)
    assert extra["lirf_missing_record"].tolist() == [1., 0., 0.]
    assert prediction_offset(extra, "lirf_missing_schedule").tolist() == [1200., 0., 0.]
    for column, value in [("ADEP_mvt", "EGLL"), ("ADEP_flt", "LIRF"), ("AIRCRAFT_TYPE_mvt", "A320")]:
        modified = raw.copy()
        modified.loc[0, column] = value
        assert quality_features(modified).loc[0, "lirf_missing_record"] == 0
    raw.loc[0, "SCHED_TIME_UTC_mvt"] = pd.NaT
    assert quality_features(raw).loc[0, "conditional_schedule_offset_s"] == 0


def test_day_offset_and_modulo_keep_signed_original_interval(movements):
    raw = raw_example(movements)
    raw.loc[0, "SCHED_TIME_UTC_mvt"] = raw.loc[0, TIME] - pd.Timedelta(seconds=90000)
    result = quality_features(raw).iloc[0]
    assert result["conditional_schedule_offset_s"] == 90000
    assert result["schedule_delta_modday_s"] == 3600
    assert result["schedule_clock_delta_s"] == -82800
    assert result["schedule_day_offset"] == 2
    raw.loc[0, "SCHED_TIME_UTC_mvt"] = raw.loc[0, TIME] + pd.Timedelta(seconds=60)
    assert quality_features(raw).loc[0, "conditional_schedule_offset_s"] == -60


def test_quality_is_flight_shape_not_full_identifier(movements):
    raw = raw_example(movements)
    result = quality_features(raw)
    assert result.loc[0, "record_flight_prefix"] == "ITY"
    assert result.loc[0, "flight_prefix_length"] == 5
    assert result.loc[0, "flight_prefix_tail_repeat"] == 1
    assert result.loc[0, "flight_standard_shape"] == 0
    assert result.loc[1, "flight_standard_shape"] == 1
    assert "ITYTY630" not in result.astype("string").to_numpy()
    assert "FLIGHT_ID_mvt" not in FEATURE_NAMES and ID not in FEATURE_NAMES


def test_audit_features_and_refit_are_rejected(movements, protocol):
    raw = raw_example(movements)
    raw.loc[0, TIME] = pd.Timestamp("2025-12-01", tz="UTC")
    with pytest.raises(ValueError, match="Reserved audit"):
        quality_features(raw)
    current = configuration()
    validate_config(current, protocol)
    current["refit_months"].append("2025-12")
    with pytest.raises(ValueError, match="reserved audit"):
        validate_config(current, protocol)


def test_portable_parent_ignores_only_measured_metadata():
    original = {"created_at_utc": "old", "preparation": {"hostname": "A"},
                "sources": [{"sha256": "same", "wall_seconds": 1.0}],
                "artifacts": [{"sha256": "data"}], "folds": {"score_ids": "frozen"}}
    other = deepcopy(original)
    other.update(created_at_utc="new", preparation={"hostname": "B"})
    other["sources"][0]["wall_seconds"] = 23.0
    assert scientific_parent(original) == scientific_parent(other)
    assert original["sources"][0]["wall_seconds"] == 1.0
    other["artifacts"][0]["sha256"] = "changed data"
    assert scientific_parent(original) != scientific_parent(other)
    other = deepcopy(original)
    other["folds"]["score_ids"] = "changed scoring IDs"
    assert scientific_parent(original) != scientific_parent(other)


def test_phase4_encoder_is_fit_only_and_rejects_target_columns():
    features = sample_features(10)
    encoder = fit_encoder(features)
    changed = features.copy()
    changed["record_flight_prefix"] = "unseen"
    assert transform(changed, encoder)["record_flight_prefix"].isna().all()
    assert "unseen" not in encoder["vocabularies"]["record_flight_prefix"]
    changed[TARGET] = 5
    with pytest.raises(ValueError, match="feature contract"):
        transform(changed, encoder)


def test_offset_algebra_and_id_alignment_preserve_unusual_labels():
    features = sample_features(4)
    truth = pd.DataFrame({ID: [3., 2., 1., 0.], TARGET: [-500., 150000., 0., 800.]})
    y = aligned_target(features, truth)
    offset = prediction_offset(features, "lirf_missing_schedule")
    residual_prediction = np.array([15., 100., -25., 400.])
    np.testing.assert_array_equal((residual_prediction-(y-offset))**2, ((residual_prediction+offset)-y)**2)
    assert y.tolist() == [800., 0., 150000., -500.]
    truth.loc[0, ID] = 100
    with pytest.raises(ValueError, match="Unaligned"):
        aligned_target(features, truth)


def test_airport_routes_are_disjoint_and_complete():
    features = sample_features(8)
    egll, lirf = route_mask(features, "airport:EGLL"), route_mask(features, "airport:LIRF")
    assert not (egll & lirf).any() and (egll | lirf).all()
    np.testing.assert_array_equal(route_mask(features, "lirf_missing_record"), lirf)
    with pytest.raises(ValueError, match="Unknown"):
        route_mask(features, "target > 10800")


def test_native_residual_pipeline_extrapolates_and_reloads(tmp_path):
    frame = sample_features()
    truth = pd.DataFrame({ID: frame[ID], TARGET: prediction_offset(frame, "lirf_missing_schedule")
                         + 900 + 100*frame["airport"].eq("LIRF").to_numpy()})
    current = configuration()
    settings = {k: current[k] for k in ["threads", "seed", "lightgbm"]}
    settings["threads"] = 2
    settings["lightgbm"].update(rounds=60, early_stopping_rounds=10)
    settings["lightgbm"]["parameters"].update(learning_rate=0.2, num_leaves=7, min_data_in_leaf=10)
    node = fit_node(tmp_path, "global", "phase4", "lirf_missing_schedule", "all", settings,
                    frame.iloc[:320], truth.iloc[:320], frame.iloc[320:], truth.iloc[320:])
    pipeline = {"default": "global", "nodes": {"global": node}, "patches": []}
    values = pipeline_prediction(tmp_path, pipeline, frame.iloc[320:], 2)
    assert np.sqrt(np.mean((values-truth.iloc[320:][TARGET].to_numpy())**2)) < 10
    np.testing.assert_allclose(values, pipeline_prediction(tmp_path, json.loads(json.dumps(pipeline)), frame.iloc[320:], 2), atol=1e-9, rtol=0)
    pipeline["patches"] = [{"kind": "schedule", "route": "lirf_missing_record", "weight": 1.0}]
    patched = pipeline_prediction(tmp_path, pipeline, frame.iloc[320:], 2)
    mask = route_mask(frame.iloc[320:], "lirf_missing_record")
    np.testing.assert_array_equal(patched[mask], frame.iloc[320:].loc[mask, "schedule_to_movement_s"])
    np.testing.assert_array_equal(patched[~mask], values[~mask])
    with pytest.raises(ValueError, match="fixed refit"):
        fit_node(tmp_path, "bad", "phase4", "none", "all", settings,
                 frame.iloc[:320], truth.iloc[:320], frame.iloc[320:], truth.iloc[320:], iterations=5)
