"""Test the physical loss identity, complete routes and portable scaled native models."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from test_ladder import model_fixture
from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash
from prc2026_ladder_scale.targets import offset_scale, target_and_weight
from prc2026_ladder_scale.native import fit_node
from prc2026_ladder_scale.inference import node_prediction, pipeline_prediction, recipe_pipeline, compose, route_mask
from prc2026_ladder_scale.training import validate_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from verify_ladder_scale import independent_node_values


def config():
    return json.loads((ROOT/"configs/ladder_scale_v1.json").read_text())


@pytest.mark.parametrize("name", ["clock_scale60", "clock_scale300", "missing_scale300", "missing_scale900"])
def test_scaled_loss_equals_full_squared_error_in_seconds(name):
    variant = config()["variants"][name]
    frame, truth = model_fixture(variant["route"])
    frame["nmclock_clock_difference_s"] = np.linspace(-5000, 5000, len(frame))
    frame.loc[0, "nmclock_clock_difference_s"] = np.nan
    frame["schedule_to_movement_s"] = np.linspace(-86400, 120000, len(frame))
    frame.loc[frame["airport"].eq("LIRF"), "lirf_missing_record"] = 1.
    offset, scale = offset_scale(frame, variant)
    z, weights = target_and_weight(frame, truth[TARGET], variant)
    estimate = np.linspace(-2, 2, len(frame))
    physical = truth[TARGET].to_numpy()-offset-scale*estimate
    np.testing.assert_allclose(weights*np.square(z-estimate)*300**2, np.square(physical), rtol=2e-14, atol=1e-6)
    np.testing.assert_allclose(offset+scale*z, truth[TARGET], rtol=1e-12, atol=1e-9)
    assert len(z) == len(truth) and truth[TARGET].min() < 0 and truth[TARGET].max() > 86400
    assert np.isfinite(weights).all() and (weights > 0).all()


@pytest.mark.parametrize("name", ["clock_scale60", "clock_scale300", "missing_scale300", "missing_scale900"])
def test_scaled_native_reload_and_october_metric_retain_every_target(tmp_path, name):
    variant = deepcopy(config()["variants"][name])
    variant["settings"]["threads"] = 1
    variant["settings"]["lightgbm"].update(rounds=12, early_stopping_rounds=3)
    variant["settings"]["lightgbm"]["parameters"].update(num_leaves=7, min_data_in_leaf=10)
    frame, truth = model_fixture(variant["route"])
    frame["nmclock_clock_difference_s"] = np.linspace(-3000, 3000, len(frame))
    node = fit_node(tmp_path, "scaled", variant, frame.iloc[:180], truth.iloc[:180].iloc[::-1],
                    frame.iloc[180:], truth.iloc[180:])
    assert node["fit_rows"] == 180 and node["fit_ids_sha256"] == identity_hash(frame.iloc[:180][ID])
    assert node["target_filter"] == "none" and node["target_transform"] == variant["target_transform"]
    values = node_prediction(tmp_path, node, frame.iloc[180:], 1)
    np.testing.assert_array_equal(values, independent_node_values(tmp_path, node, frame.iloc[180:], 1))
    actual = float(np.sqrt(np.mean((values-truth.iloc[180:][TARGET].to_numpy())**2)))
    assert actual == pytest.approx(node["early_stop_rmse_seconds"], rel=1e-12)
    assert node["max_model_reload_difference_seconds"] == 0
    history = json.loads((tmp_path/node["history_path"]).read_text())
    assert set(history["early_stop"]) == {"rmse_seconds"}
    card = fit_node(tmp_path, "fixed", variant, frame, truth, iterations=node["iterations"])
    assert card["iterations"] == node["iterations"] and card["early_stop_rows"] == 0
    assert card["fit_rows"] == len(truth)


@pytest.mark.parametrize("weight", [.25, .75, 1.])
def test_scaled_portable_recipe_matches_complete_saved_prediction_math(monkeypatch, weight):
    import prc2026_ladder_scale.inference as inference
    features = pd.DataFrame({ID: [0., 1., 2., 3.], "nmclock_available": [1., 0., 1., 0.],
                             "nmclock_proxy_s": [900., np.nan, -60., np.nan]})
    base = np.array([-10.2, 120000.1, 1300.3, 500.4])
    replacements = {"present": base+100, "missing": base-50}
    monkeypatch.setattr(inference, "prior_prediction",
                        lambda run, pipe, frame, threads: base[frame[ID].astype(int)])
    def fake_node(run, node, frame, threads):
        assert route_mask(frame, node["route"]).all()
        return replacements[node["name"]][frame[ID].astype(int)]
    monkeypatch.setattr(inference, "node_prediction", fake_node)
    sources = {"v6": {"kind": "frozen_nmclock", "pipeline": {}},
               "present": {"kind": "scaled_node", "node": {"route": "clock_available", "name": "present"}},
               "missing": {"kind": "scaled_node", "node": {"route": "clock_missing", "name": "missing"}}}
    family = {"clock_available": {"present": 1.}, "clock_missing": {"missing": 1.}}
    actual = pipeline_prediction(Path("."), recipe_pipeline(family, weight, sources), features, 1)
    expected = compose(base, replacements, np.array([True, False, True, False]), family, weight)
    np.testing.assert_array_equal(actual, expected)
    family["clock_missing"] = {"v6": 1.}
    actual = pipeline_prediction(Path("."), recipe_pipeline(family, weight, sources), features, 1)
    np.testing.assert_array_equal(actual[[1, 3]], base[[1, 3]])


def test_scale_declaration_rejects_changed_loss_unit_and_undeclared_routes():
    original = config()
    protocol = json.loads((ROOT/"configs/validation_v1.json").read_text())
    validate_config(original, protocol)
    for key, value in [("audit_used", True), ("threads", 32), ("target_filter", "positive_only")]:
        bad = deepcopy(original)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_config(bad, protocol)
    for key, value in [("weight_unit_seconds", 600.), ("minimum_seconds", 1.)]:
        bad = deepcopy(original)
        bad["variants"]["clock_scale60"]["target_transform"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad, protocol)
