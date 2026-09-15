"""Promotion for the follow-up must beat the preceding incumbent on both complete folds."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from prc2026.artifacts import object_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from select_ladder_scale import eligible_recipe
from verify_ladder_scale import compare_selection


def fixture():
    config = json.loads((ROOT/"configs/ladder_scale_v1.json").read_text())
    family = "clock300_missing300"
    metrics = {fold: {"overall": {"rows": 20, "rmse_seconds": 280.}}
               for fold in ["seasonal", "forward"]}
    selection = {
        "audit_used": False, "selected_candidate": "chosen", "config_sha256": object_hash(config),
        "bundle_id": "bundle", "model_code_sha256": "code",
        "reference_metrics": {fold: {"overall": {"rows": 20, "rmse_seconds": 300.}}
                              for fold in ["seasonal", "forward"]},
        "promotion_reference": {"name": "preceding ladder winner", "metrics": metrics},
        "development_runs": {name: {"seasonal": {"iterations": 10}, "forward": {"iterations": 11}}
                             for name in ["clock_scale300", "missing_scale300"]},
        "refit_iterations": {"clock_scale300": 10, "missing_scale300": 10},
        "candidates": {"chosen": {"family": family, "routes": config["candidate_families"][family],
                        "new_weight": .75, "eligible": True,
                        "folds": {fold: {"metrics": {"overall": {"rows": 20, "rmse_seconds": 270.}}}
                                  for fold in ["seasonal", "forward"]}}},
    }
    return config, selection, {"bundle_id": "bundle"}, {"sha256": "code"}


def test_beating_v6_alone_cannot_replace_a_better_preceding_incumbent():
    config, selected, manifest, code = fixture()
    assert eligible_recipe(config, selected, manifest, code)["new_weight"] == .75
    bad = deepcopy(selected)
    bad["candidates"]["chosen"]["folds"]["seasonal"]["metrics"]["overall"]["rmse_seconds"] = 290.
    with pytest.raises(ValueError, match="two-fold"):
        eligible_recipe(config, bad, manifest, code)
    bad = deepcopy(selected)
    bad["candidates"]["chosen"]["folds"]["forward"]["metrics"]["overall"]["rows"] = 19
    with pytest.raises(ValueError, match="two-fold"):
        eligible_recipe(config, bad, manifest, code)


def test_verification_rejects_a_changed_promotion_reference():
    fields = ["config_sha256", "model_code_sha256", "selection_code_sha256", "paired_day_code_sha256",
              "bundle_id", "parent_bundle_id", "candidate_count", "selected_candidate", "refit_iterations"]
    rebuilt = {name: "same" for name in fields}
    rebuilt.update(candidates={}, reference_metrics={}, selected_paired_days={},
                   promotion_reference={"name": "incumbent", "metrics": {"by_month": {1: {"rmse": 10.}}}})
    saved = json.loads(json.dumps(rebuilt))
    compare_selection(saved, rebuilt)
    saved["promotion_reference"]["metrics"]["by_month"]["1"]["rmse"] = 11.
    with pytest.raises(ValueError):
        compare_selection(saved, rebuilt)

