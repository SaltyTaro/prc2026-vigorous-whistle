"""Refit must not proceed on a tampered or one-fold-only improvement."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
from prc2026.artifacts import object_hash

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weather_refit_tool", root/"tools/refit_weather.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    config = json.loads((root/"configs/weather_v1.json").read_text())
    selection = {"audit_used": False, "selected_candidate": "example", "config_sha256": object_hash(config),
                 "bundle_id": "bundle", "model_code_sha256": "code", "refit_iterations": {"cat_weather": 4500},
                 "development_runs": {"cat_weather": {"seasonal": {"iterations": 4000}, "forward": {"iterations": 5000}}},
                 "reference_metrics": {f: {"overall": {"rmse_seconds": 300.}} for f in ["seasonal", "forward"]},
                 "candidates": {"example": {"eligible": True, "variant": "cat_weather", "family_weights": {"cat_weather": 1.},
                                             "new_weight": .5, "folds": {f: {"metrics": {"overall": {"rmse_seconds": 295.}}} for f in ["seasonal", "forward"]}}}}
    return config, selection, {"bundle_id": "bundle"}, {"sha256": "code"}


def test_refit_requires_actual_both_fold_margin_even_if_eligible_flag_is_true():
    config, selection, bundle, code = fixture()
    assert module.eligible_recipe(config, selection, bundle, code)["new_weight"] == .5
    for value in [299., float("nan")]:
        changed = deepcopy(selection)
        changed["candidates"]["example"]["folds"]["seasonal"]["metrics"]["overall"]["rmse_seconds"] = value
        with pytest.raises(ValueError, match="two-fold"):
            module.eligible_recipe(config, changed, bundle, code)


def test_refit_rejects_changed_code_undeclared_weights_and_altered_iterations():
    config, selection, bundle, code = fixture()
    changed = deepcopy(selection)
    changed["model_code_sha256"] = "changed"
    with pytest.raises(ValueError, match="unchanged"):
        module.eligible_recipe(config, changed, bundle, code)
    changed = deepcopy(selection)
    changed["candidates"]["example"]["new_weight"] = .33
    with pytest.raises(ValueError, match="Undeclared"):
        module.eligible_recipe(config, changed, bundle, code)
    changed = deepcopy(selection)
    changed["refit_iterations"]["cat_weather"] = 6000
    with pytest.raises(ValueError, match="medians"):
        module.eligible_recipe(config, changed, bundle, code)
