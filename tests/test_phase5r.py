from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc2026.artifacts import write_json
from prc2026.features import BLOCK, ID, PHASE, TARGET, TIME
from prc2026_models.experiments import receipts
from prc2026_phase4.features import prediction_offset
from prc2026_phase5r.baseline import apply_map, center_scale, cross_fit, fit_map, group_folds
from prc2026_phase5r.encoding import assemble, fit_encoder, transform
from prc2026_phase5r.features import CATEGORICAL_COLUMNS, FEATURE_NAMES, GROUND_NAMES, INPUT_COLUMNS, ground_features
from prc2026_phase5r.inference import copy_legacy_pipeline, node_prediction, pipeline_prediction
from prc2026_phase5r.training import fit_node, settings_for, validate_config


def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/phase5r_v1.json").read_text())


def context():
    times = pd.to_datetime(["2025-03-05T09:30Z", "2025-03-05T09:50Z", "2025-03-05T10:00Z",
                           "2025-03-05T10:00Z", "2025-03-05T10:20Z", "2025-03-05T10:10Z"])
    frame = pd.DataFrame({ID: np.arange(1, 7, dtype=float), PHASE: ["DEP", "ARR", "DEP", "DEP", "DEP", "ARR"],
                          TIME: times, "ADEP_mvt": ["EGLL", "KJFK", "EGLL", "EGLL", "EGLL", "KJFK"],
                          "ADES_mvt": ["KJFK", "EGLL", "KJFK", "KJFK", "KJFK", "EGLL"],
                          "RUNWAY_mvt": ["27R"]*6, "STAND_mvt": ["A1"]*6, "WK_TBL_CAT_flt": ["M", "H", "M", "M", "M", "M"]})
    frame["SCHED_TIME_UTC_mvt"] = pd.to_datetime(["2025-03-05T09:05Z", "2025-03-05T09:10Z", "2025-03-05T09:20Z",
                                               "2025-03-05T09:45Z", "2025-03-05T09:40Z", "2025-03-05T10:00Z"])
    frame["EOBT_1_flt"] = pd.to_datetime(["2025-03-05T09:15Z", "2025-03-05T07:00Z", "2025-03-05T09:40Z",
                                        "2025-03-05T09:50Z", "2025-03-05T09:55Z", "2025-03-05T07:00Z"])
    return frame


def sample(n=360):
    rng = np.random.default_rng(120)
    data = {ID: np.arange(n, dtype=float)}
    for name in FEATURE_NAMES:
        data[name] = rng.choice(["A", "B"], n) if name in CATEGORICAL_COLUMNS else rng.normal(size=n).astype("float32")
    frame = pd.DataFrame(data)
    frame["airport"] = np.where(np.arange(n) % 2 == 0, "LIRF", "EGLL")
    frame["nm_missing"] = (np.arange(n) % 3 == 0).astype("float32")
    frame["aircraft_missing"] = frame["nm_missing"]
    frame["lirf_missing_record"] = (frame["airport"].eq("LIRF") & frame["nm_missing"].eq(1)).astype("float32")
    frame["conditional_schedule_offset_s"] = frame["lirf_missing_record"]*90000
    y = prediction_offset(frame, "lirf_missing_schedule")+800+200*frame["airport"].eq("LIRF").to_numpy()+30*frame["schedule_to_movement_s"].to_numpy()
    truth = pd.DataFrame({ID: frame[ID], TARGET: y, "FLIGHT_ID_mvt": (np.arange(n)//2).astype(float)})
    return frame, truth


def test_demand_and_service_have_separate_clocks_and_exclude_focal_plan():
    raw = context()
    out = ground_features(raw, pd.Series([4.]))
    assert out["gd_0_schedule_count_30m"].iloc[0] == 1
    assert out["gd_0_eobt_count_30m"].iloc[0] == 2
    assert out["service_previous_phase"].iloc[0] == "ARR"
    assert out["service_wake_pair"].iloc[0] == "H|M"
    assert out["gs_10_arr_count_5m"].iloc[0] == 0  # exactly at the earlier query
    assert out["service_airport_all_gap5_mean_s"].iloc[0] == 1200
    changed = raw.copy()
    changed.loc[changed[ID].isin([3., 5., 6.]), TIME] += pd.Timedelta(hours=2)
    pd.testing.assert_frame_equal(out, ground_features(changed, pd.Series([4.])), check_exact=True)
    changed = raw.copy()
    changed.loc[changed[ID].eq(5.), "SCHED_TIME_UTC_mvt"] = pd.Timestamp("2025-03-05T10:01Z")
    changed_out = ground_features(changed, pd.Series([4.]))
    assert changed_out["gd_0_schedule_count_30m"].iloc[0] == 0
    assert changed_out["service_previous_phase"].iloc[0] == "ARR"


def test_ground_context_ignores_all_forbidden_columns_and_row_order():
    raw = context()
    expected = ground_features(raw, pd.Series([4., 1.]))
    assert len(GROUND_NAMES) == len(set(GROUND_NAMES))
    forbidden = [TARGET, BLOCK, "AOBT_3_flt", "LOBT_flt", "ARVT_3_flt"]
    assert not set(forbidden) & set(INPUT_COLUMNS)
    for name in forbidden:
        raw[name] = 90000
    pd.testing.assert_frame_equal(expected, ground_features(raw.sample(frac=1, random_state=8), pd.Series([4., 1.])), check_exact=True)
    raw.loc[0, TIME] = pd.Timestamp("2025-12-01T00:00Z")
    with pytest.raises(ValueError, match="Reserved audit"):
        ground_features(raw, pd.Series([4.]))


def test_crossfit_excludes_entire_flight_group_and_keeps_all_labels():
    features, truth = sample(120)
    spec = config()["baseline"]
    folds = group_folds(features, truth, 5, spec["seed"])
    assert np.all(folds[::2] == folds[1::2])
    original, model, receipt, assignments = cross_fit(features, truth, spec)
    changed = truth.copy()
    changed.loc[:1, TARGET] = [-50000, 180000]
    alternate, _, _, _ = cross_fit(features, changed, spec)
    pd.testing.assert_frame_equal(original.iloc[:2], alternate.iloc[:2], check_exact=True)
    assert model["fit_rows"] == len(features)
    assert sum(r["heldout_rows"] for r in receipt["folds"]) == len(features)
    assert len(assignments) == len(features)
    assert receipt["all_fitting_labels_retained"]


def test_unseen_baseline_groups_back_off_without_consuming_query_labels():
    features, truth = sample(120)
    residual = truth[TARGET].to_numpy()-prediction_offset(features, "lirf_missing_schedule")
    model = fit_map(features, residual, config()["baseline"])
    unknown = features.iloc[:3].copy()
    unknown["airport"] = "NEW"
    result = apply_map(unknown, json.loads(json.dumps(model)))
    assert np.isfinite(result.to_numpy()).all()
    assert result["taxi_base_support"].eq(0).all()
    np.testing.assert_allclose(result["taxi_base_q20_s"], np.quantile(residual, .2), rtol=1e-6)


def test_scaled_target_preserves_squared_seconds_loss_with_extreme_and_negative_labels():
    baseline = pd.DataFrame({"taxi_base_q20_s": [300, -50, 1200], "taxi_airport_scale_s": [60, 350, 1200]})
    center, scale, weights = center_scale(baseline, "baseline_scaled", 3)
    truth = np.array([-200., 90000., 400.])
    offset = np.array([0., 86400., 0.])
    standardized_truth = (truth-offset-center)/scale
    prediction_z = np.array([-.5, 3., 8.])
    prediction_seconds = prediction_z*scale+center+offset
    np.testing.assert_allclose(weights*(prediction_z-standardized_truth)**2,
                               (prediction_seconds-truth)**2/600**2, rtol=1e-13)


def test_new_encoder_has_fit_only_categories_and_explicit_contract():
    features, _ = sample(12)
    spec = config()["variants"]["demand_service"]
    frame = assemble(features, spec)
    encoder = fit_encoder(frame, spec)
    frame["service_wake_pair"] = "NEW"
    assert transform(frame, encoder)["service_wake_pair"].eq("__MISSING__").all()
    frame[TARGET] = 800
    with pytest.raises(ValueError, match="contract"):
        transform(frame, encoder)


@pytest.mark.parametrize("target", ["none", "baseline_scaled"])
def test_native_new_model_transform_and_pipeline_roundtrip(tmp_path, target):
    features, truth = sample()
    current = config()
    current["threads"] = 2
    spec = deepcopy(current["variants"]["baseline_scaled" if target != "none" else "demand_service"])
    settings = settings_for(current, spec)
    settings["lightgbm"].update(rounds=50, early_stopping_rounds=8)
    settings["lightgbm"]["parameters"].update(num_leaves=7, min_data_in_leaf=5, lambda_l2=0.01, learning_rate=.15)
    node = fit_node(tmp_path, "new", spec, settings, features.iloc[:280], truth.iloc[:280], features.iloc[280:], truth.iloc[280:])
    expected = node_prediction(tmp_path, node, features.iloc[280:], 2)
    pipeline = {"kind": "new_node", "node": node}
    np.testing.assert_allclose(pipeline_prediction(tmp_path, json.loads(json.dumps(pipeline)), features.iloc[280:], 2), expected, rtol=0, atol=1e-9)
    assert np.sqrt(np.mean((expected-truth[TARGET].to_numpy()[280:])**2)) < 45
    assert node["max_model_reload_difference_seconds"] == 0
    with pytest.raises(ValueError, match="fixed refit"):
        fit_node(tmp_path, "invalid", spec, settings, features.iloc[:280], truth.iloc[:280], features.iloc[280:], truth.iloc[280:], iterations=2)


def test_copying_shared_frozen_models_keeps_one_file_and_refuses_overwrite(tmp_path):
    source, dest = tmp_path/"source", tmp_path/"destination"
    source.mkdir()
    dest.mkdir()
    (source/"shared.txt").write_text("native model placeholder", encoding="utf-8")
    pipeline = {"kind": "mixture", "members": [{"name": name, "weight": .5, "pipeline": {
        "default": "global", "nodes": {"global": {"model": "shared.txt"}}, "patches": []}} for name in ["a", "b"]]}
    write_json(source/"pipeline.json", pipeline)
    write_json(source/"run.json", {"artifacts": receipts(source)})
    copied = copy_legacy_pipeline(dest, source, "old")
    assert len(list(dest.glob("*.txt"))) == 1
    assert copied["members"][0]["pipeline"]["nodes"]["global"]["model"] == "old_shared.txt"
    assert copied["members"][1]["pipeline"]["nodes"]["global"]["model"] == "old_shared.txt"
    with pytest.raises(ValueError, match="overwrite"):
        copy_legacy_pipeline(dest, source, "old")


def test_new_config_protects_audit_and_both_outer_folds(protocol):
    current = config()
    validate_config(current, protocol)
    current["refit_months"].append("2025-12")
    with pytest.raises(ValueError, match="reserved audit"):
        validate_config(current, protocol)
