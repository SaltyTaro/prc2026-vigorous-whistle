from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, TARGET
from prc2026_phase4.features import CATEGORICAL_COLUMNS, FEATURE_NAMES, prediction_offset
from prc2026_phase5.inference import (cat_transform, combine_predictions, fit_cat_encoder,
                                     pipeline_prediction, validate_weights)
from prc2026_phase5.training import derive_settings, fit_component, validate_config


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/phase5_v1.json").read_text())


def sample(n=280):
    rng = np.random.default_rng(20260913)
    data = {ID: np.arange(n, dtype=float)}
    for name in FEATURE_NAMES:
        data[name] = rng.choice(["A", "B"], n) if name in CATEGORICAL_COLUMNS else rng.normal(size=n).astype("float32")
    frame = pd.DataFrame(data)
    frame["airport"] = np.where(np.arange(n) % 2 == 0, "LIRF", "EGLL")
    frame["lirf_missing_record"] = frame["airport"].eq("LIRF").astype("float32")
    frame["conditional_schedule_offset_s"] = frame["lirf_missing_record"]*np.arange(n)*400
    return frame


def test_sorted_receipts_keep_explicit_execution_order_and_audit_guard(protocol):
    current = json.loads(json.dumps(config(), sort_keys=True))
    validate_config(current, protocol)
    current["refit_months"].append("2025-12")
    with pytest.raises(ValueError, match="reserved audit"):
        validate_config(current, protocol)


def test_component_overrides_do_not_mutate_reference_settings():
    original = {"settings": {"threads": 16, "seed": 20260913,
                            "lightgbm": {"rounds": 1800, "early_stopping_rounds": 120,
                                         "parameters": {"num_leaves": 63, "lambda_l2": 10.0}}}}
    saved = deepcopy(original)
    current = config()
    settings = derive_settings(current, original, current["variants"]["global_capacity"])
    assert original == saved
    assert settings["lightgbm"]["parameters"]["num_leaves"] == 127
    assert settings["lightgbm"]["rounds"] == 3200


def test_catboost_encoder_is_fit_only_and_rejects_label_input():
    features = sample(12)
    encoder = fit_cat_encoder(features)
    changed = features.copy()
    changed["record_flight_prefix"] = "unseen operator"
    assert cat_transform(changed, encoder)["record_flight_prefix"].eq("__MISSING__").all()
    assert "unseen operator" not in encoder["vocabularies"]["record_flight_prefix"]
    changed[TARGET] = 30
    with pytest.raises(ValueError, match="feature contract"):
        cat_transform(changed, encoder)


@pytest.mark.parametrize("weights", [[-0.1, 1.1], [0.5, 0.4], [float("nan"), 1], []])
def test_invalid_ensemble_weights_fail(weights):
    with pytest.raises(ValueError, match="weights"):
        validate_weights(weights)


def test_blending_aligns_ids_and_preserves_negative_extreme_predictions():
    first = pd.DataFrame({ID: [2., 1., 3.], "prediction": [-100., 90000., 300.]})
    second = pd.DataFrame({ID: [3., 2., 1.], "prediction": [500., -300., 130000.]})
    result = combine_predictions({"a": first, "b": second}, {"a": 0.25, "b": 0.75})
    assert result[ID].tolist() == [2., 1., 3.]
    np.testing.assert_array_equal(result["prediction"], [-250., 120000., 450.])
    with pytest.raises(ValueError, match="exact same IDs"):
        combine_predictions({"a": first, "b": second.iloc[:2]}, {"a": .5, "b": .5})
    duplicate = second.copy()
    duplicate[ID] = [2., 2., 1.]
    with pytest.raises(ValueError, match="exact same IDs"):
        combine_predictions({"a": first, "b": duplicate}, {"a": .5, "b": .5})


def test_native_catboost_and_mixed_pipeline_reload(tmp_path):
    features = sample()
    y = prediction_offset(features, "lirf_missing_schedule") + 800 + 300*features["airport"].eq("LIRF").to_numpy()
    truth = pd.DataFrame({ID: features[ID], TARGET: y})
    current = config()
    spec = deepcopy(current["variants"]["catboost_global"])
    spec.update(rounds=50, early_stopping_rounds=8)
    spec["parameters"].update(depth=4, learning_rate=0.2, l2_leaf_reg=1.0)
    current["threads"] = 2
    cat_settings = derive_settings(current, {}, spec)
    blueprint = {"schema": "phase4", "route": "all", "offset": "lirf_missing_schedule", "model_kind": "catboost"}
    cat = fit_component(tmp_path, "cat", blueprint, cat_settings, features.iloc[:220], truth.iloc[:220],
                        features.iloc[220:], truth.iloc[220:])
    previous = json.loads((Path(__file__).resolve().parents[1] / "configs/phase4_v1.json").read_text())
    lgb_settings = {k: previous[k] for k in ["threads", "seed", "lightgbm"]}
    lgb_settings["threads"] = 2
    lgb_settings["lightgbm"].update(rounds=40, early_stopping_rounds=8)
    lgb_settings["lightgbm"]["parameters"].update(num_leaves=7, min_data_in_leaf=5, learning_rate=.2)
    lgb_blueprint = dict(blueprint, model_kind="lightgbm")
    lgb = fit_component(tmp_path, "lgb", lgb_blueprint, lgb_settings, features.iloc[:220], truth.iloc[:220],
                        features.iloc[220:], truth.iloc[220:])
    a = {"default": "global", "nodes": {"global": cat}, "patches": []}
    b = {"default": "global", "nodes": {"global": lgb}, "patches": []}
    mixture = {"kind": "mixture", "members": [{"name": "cat", "weight": .25, "pipeline": a},
                                               {"name": "lgb", "weight": .75, "pipeline": b}]}
    holdout = features.iloc[220:]
    values = pipeline_prediction(tmp_path, mixture, holdout, 2)
    expected = .25*pipeline_prediction(tmp_path, a, holdout, 2) + .75*pipeline_prediction(tmp_path, b, holdout, 2)
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(values, pipeline_prediction(tmp_path, json.loads(json.dumps(mixture)), holdout, 2), rtol=0, atol=1e-9)
    assert np.sqrt(np.mean((values-y[220:])**2)) < 30
    with pytest.raises(ValueError, match="fixed refit"):
        fit_component(tmp_path, "invalid", blueprint, cat_settings, features.iloc[:220], truth.iloc[:220],
                       holdout, truth.iloc[220:], iterations=3)


def test_metric_receipts_allow_json_month_keys_but_reject_changed_values(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from verify_phase5 import same_metrics

    expected = {"ranking_month_weights": {"1": 0.4, "7": 0.6}}
    same_metrics({"ranking_month_weights": {1: 0.4, 7: 0.6}}, expected)
    with pytest.raises(ValueError, match="Metric differs"):
        same_metrics({"ranking_month_weights": {1: 0.5, 7: 0.5}}, expected)
    with pytest.raises(ValueError, match="Metric keys differ"):
        same_metrics({"ranking_month_weights": {1: 0.4, "1": 0.4, 7: 0.6}}, expected)
