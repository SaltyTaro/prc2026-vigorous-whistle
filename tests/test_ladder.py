"""The ladder campaign must preserve complete populations and declared routes."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash
from prc2026_nmclock.features import FEATURE_NAMES
from prc2026_weather.features import CATEGORICAL_COLUMNS
from prc2026_ladder.inference import compose, node_prediction, pipeline_prediction, recipe_pipeline, route_mask
from prc2026_ladder.training import aligned_predictions, fit_variant, validate_config

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/"configs/ladder_v1.json").read_text())


def model_fixture(route):
    rng = np.random.default_rng(75)
    count = 240
    frame = pd.DataFrame({ID: np.arange(count, dtype=float), **{
        name: rng.choice(["A", "B"], count) if name in CATEGORICAL_COLUMNS else rng.normal(size=count)
        for name in FEATURE_NAMES}})
    frame["airport"] = np.where(np.arange(count) % 2, "LIRF", "LFPG")
    frame["lirf_missing_record"] = 0.
    frame["schedule_to_movement_s"] = 1200.
    frame["nmclock_available"] = float(route == "clock_available")
    frame["nmclock_proxy_s"] = np.linspace(-600, 3000, count) if route == "clock_available" else np.nan
    target = 1000 + 60*frame["wx_speed_mps"].to_numpy()
    target[0], target[1] = -100., 120000.
    return frame, pd.DataFrame({ID: frame[ID], TARGET: target})


@pytest.mark.parametrize("name", ["cat_missing_core", "cat_missing_ground", "lgb_missing_regularized", "cat_long"])
def test_native_routed_experts_keep_signed_and_extreme_labels(tmp_path, name):
    variant = deepcopy(config()["variants"][name])
    variant["settings"]["threads"] = 1
    kind = variant["kind"]
    variant["settings"][kind].update(rounds=12, early_stopping_rounds=3)
    variant["settings"][kind]["parameters"].update(
        {"depth": 3} if kind == "catboost" else {"num_leaves": 7, "min_data_in_leaf": 10})
    frame, truth = model_fixture(variant["route"])
    node = fit_variant(tmp_path, "expert", variant, frame.iloc[:180], truth.iloc[:180].iloc[::-1],
                       frame.iloc[180:], truth.iloc[180:])
    assert node["fit_rows"] == 180
    assert node["fit_ids_sha256"] == identity_hash(frame.iloc[:180][ID])
    assert node["target_filter"] == "none"
    assert node["max_model_reload_difference_seconds"] == 0
    values = node_prediction(tmp_path, node, frame.iloc[180:], 1)
    assert len(values) == 60 and np.isfinite(values).all()
    encoder = json.loads((tmp_path/node["encoder_path"]).read_text())
    assert ID not in encoder["feature_names"] and TARGET not in encoder["feature_names"]
    wrong = frame.copy()
    if variant["route"] == "clock_missing":
        wrong.loc[0, ["nmclock_available", "nmclock_proxy_s"]] = [1., 900.]
    else:
        wrong.loc[0, ["nmclock_available", "nmclock_proxy_s"]] = [0., np.nan]
    with pytest.raises(ValueError, match="declared nonempty route"):
        fit_variant(tmp_path, "wrong", variant, wrong, truth)


@pytest.mark.parametrize("weight", [.5, .75, 1.])
def test_recipe_math_covers_both_routes_and_preserves_untouched_rows(weight):
    base = np.array([-10.2, 120000.1, 1300.3, 500.4])
    available = np.array([True, False, True, False])
    components = {"v6_global": base+100, "v6_airports": base-20, "cat_missing_core": base+50}
    family = {"clock_available": {"v6_global": .75, "v6_airports": .25},
              "clock_missing": {"cat_missing_core": 1.}}
    actual = compose(base, components, available, family, weight)
    expected = base + weight*np.array([70., 50., 70., 50.])
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    family["clock_available"] = {"v6": 1.}
    actual = compose(base, components, available, family, weight)
    np.testing.assert_array_equal(actual[available], base[available])
    assert actual[0] < 0 and actual[1] > 100000


@pytest.mark.parametrize("weight", [.5, .75, 1.])
def test_portable_pipeline_matches_saved_prediction_recipe(monkeypatch, weight):
    import prc2026_ladder.inference as inference
    features = pd.DataFrame({ID: [0., 1., 2., 3.], "nmclock_available": [1., 0., 1., 0.],
                             "nmclock_proxy_s": [900., np.nan, -60., np.nan]})
    base = np.array([-10.2, 120000.1, 1300.3, 500.4])
    components = {"cat_long": base+100, "cat_missing_core": base-50}
    monkeypatch.setattr(inference, "frozen_prediction",
                        lambda run, pipe, frame, threads: base[frame[ID].astype(int)])
    def fake_node(run, node, frame, threads):
        assert route_mask(frame, node["route"]).all()
        return components[node["name"]][frame[ID].astype(int)]
    monkeypatch.setattr(inference, "node_prediction", fake_node)
    sources = {"v6": {"kind": "frozen_nmclock", "pipeline": {}},
               "cat_long": {"kind": "node", "node": {"name": "cat_long", "route": "clock_available"}},
               "cat_missing_core": {"kind": "node", "node": {"name": "cat_missing_core", "route": "clock_missing"}}}
    family = {"clock_available": {"cat_long": 1.}, "clock_missing": {"cat_missing_core": 1.}}
    expected = compose(base, components, np.array([True, False, True, False]), family, weight)
    actual = pipeline_prediction(Path("."), recipe_pipeline(family, weight, sources), features, 1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    family["clock_missing"] = {"v6": 1.}
    actual = pipeline_prediction(Path("."), recipe_pipeline(family, weight, sources), features, 1)
    np.testing.assert_array_equal(actual[[1, 3]], base[[1, 3]])


def test_configuration_rejects_audit_new_features_and_wrong_routes():
    original = config()
    protocol = json.loads((ROOT/"configs/validation_v1.json").read_text())
    validate_config(original, protocol)
    for key, value in [("audit_used", True), ("target_filter", "positive_only"),
                       ("prediction_postprocess", "clip"), ("new_shared_features", True),
                       ("threads", 32), ("refit_months", original["refit_months"]+["2025-12"])]:
        bad = deepcopy(original)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_config(bad, protocol)
    bad = deepcopy(original)
    bad["candidate_families"]["missing_core"]["clock_missing"] = {"v6_global": 1.}
    with pytest.raises(ValueError, match="cannot predict missing"):
        validate_config(bad, protocol)
    bad = deepcopy(original)
    bad["feature_contracts"]["ground_weather"].append(TARGET)
    with pytest.raises(ValueError, match="feature contract"):
        validate_config(bad, protocol)


def test_complete_prediction_alignment_rejects_missing_and_duplicate_ids():
    features = pd.DataFrame({ID: [5., 2., 8.]})
    saved = pd.DataFrame({ID: [8., 5., 2.], "prediction": [-100., 120000., 500.]})
    actual = aligned_predictions(features, saved)
    assert actual["prediction"].tolist() == [120000., 500., -100.]
    with pytest.raises(ValueError, match="complete feature population"):
        aligned_predictions(features, saved.iloc[:2])
    saved.loc[0, ID] = 5.
    with pytest.raises(ValueError, match="complete feature population"):
        aligned_predictions(features, saved)


def test_recipe_rejects_invalid_weights_and_incomplete_routes():
    base = np.array([1., 2.])
    family = {"clock_available": {"v6": 1.}, "clock_missing": {"v6": 1.}}
    with pytest.raises(ValueError, match="convex"):
        compose(base, {}, np.array([True, False]), family, 1.1)
    with pytest.raises(ValueError, match="Both complementary"):
        compose(base, {}, np.array([True, False]), {"clock_available": {"v6": 1.}}, .5)
    with pytest.raises(ValueError, match="boolean routes"):
        compose(base, {}, np.array([1, 0]), family, .5)
