"""Promotion, refit population and independent-copy checks for the ladder."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from prc2026.artifacts import object_hash
from prc2026.features import ID, TIME
from select_ladder import eligible_recipe
from refit_ladder import copy_reference, refit_assignment
from verify_ladder import check_imported_sources, compare_selection


def selection_fixture():
    config = json.loads((ROOT/"configs/ladder_v1.json").read_text())
    family = "long_missing_ground"
    selection = {
        "audit_used": False, "selected_candidate": "chosen",
        "config_sha256": object_hash(config), "bundle_id": "bundle", "model_code_sha256": "code",
        "reference_metrics": {fold: {"overall": {"rmse_seconds": 300., "rows": 20}} for fold in config["development_folds"]},
        "development_runs": {
            "cat_long": {"seasonal": {"iterations": 11}, "forward": {"iterations": 14}},
            "cat_missing_ground": {"seasonal": {"iterations": 20}, "forward": {"iterations": 23}},
        },
        "refit_iterations": {"cat_long": 12, "cat_missing_ground": 22},
        "candidates": {"chosen": {
            "family": family, "routes": deepcopy(config["candidate_families"][family]), "new_weight": .75,
            "eligible": True, "folds": {fold: {"metrics": {"overall": {"rmse_seconds": 280., "rows": 20}}}
                                      for fold in config["development_folds"]},
        }},
    }
    return config, selection, {"bundle_id": "bundle"}, {"sha256": "code"}


def test_refit_rejects_an_incomplete_or_one_fold_promotion():
    config, selection, manifest, code = selection_fixture()
    assert eligible_recipe(config, selection, manifest, code)["new_weight"] == .75
    for value in [299., float("nan")]:
        bad = deepcopy(selection)
        bad["candidates"]["chosen"]["folds"]["seasonal"]["metrics"]["overall"]["rmse_seconds"] = value
        with pytest.raises(ValueError, match="two-fold"):
            eligible_recipe(config, bad, manifest, code)
    bad = deepcopy(selection)
    bad["candidates"]["chosen"]["folds"]["forward"]["metrics"]["overall"]["rows"] = 19
    with pytest.raises(ValueError, match="two-fold"):
        eligible_recipe(config, bad, manifest, code)
    bad = deepcopy(selection)
    del bad["candidates"]["chosen"]["folds"]["forward"]
    with pytest.raises(ValueError, match="Both complete"):
        eligible_recipe(config, bad, manifest, code)


def test_refit_rejects_changed_routes_weights_and_iteration_medians():
    config, selection, manifest, code = selection_fixture()
    for name in ["cat_long", "cat_missing_ground"]:
        bad = deepcopy(selection)
        bad["refit_iterations"][name] += 1
        with pytest.raises(ValueError, match="October-selected medians"):
            eligible_recipe(config, bad, manifest, code)
    bad = deepcopy(selection)
    bad["candidates"]["chosen"]["new_weight"] = .33
    with pytest.raises(ValueError, match="Undeclared"):
        eligible_recipe(config, bad, manifest, code)
    bad = deepcopy(selection)
    bad["candidates"]["chosen"]["routes"]["clock_missing"] = {"v6": 1.}
    with pytest.raises(ValueError, match="Undeclared"):
        eligible_recipe(config, bad, manifest, code)
    bad = deepcopy(selection)
    bad["model_code_sha256"] = "changed"
    with pytest.raises(ValueError, match="unchanged"):
        eligible_recipe(config, bad, manifest, code)


def test_refit_uses_actual_utc_months_and_protects_december_flight_groups():
    catalog = pd.DataFrame({
        ID: [1., 2., 3., 4., 5., 6.],
        TIME: pd.to_datetime(["2025-01-01T00:00Z", "2025-12-01T00:00Z", "2025-11-30T23:59Z",
                             "2025-10-01T00:00Z", "2025-12-31T23:59Z", "2026-01-01T00:00Z"], utc=True),
        "FLIGHT_ID_mvt": ["shared", "shared", None, "independent", None, "next_year"],
        "source_file": ["irrelevant"]*6,
    })
    assignment = refit_assignment(catalog, {"reserved_audit_months": ["2025-12"]},
                                  [f"2025-{month:02}" for month in range(1, 12)])
    assert assignment.loc[assignment.partition.eq("fit"), ID].tolist() == [3., 4.]


def test_independent_copy_check_rejects_changed_parent_native_bytes(tmp_path):
    source, run = tmp_path/"source", tmp_path/"run"
    source.mkdir()
    run.mkdir()
    global_node = {"model_path": "g.txt", "encoder_path": "g.json"}
    airports = {"TEST": {"model_path": "a.txt", "encoder_path": "a.json"}}
    for name in ["g.txt", "g.json", "a.txt", "a.json"]:
        (source/name).write_text("original "+name)
    (source/"nodes.json").write_text(json.dumps({"cat_residual": {"global": global_node, "airports": airports}}))
    (source/"pipeline.json").write_text(json.dumps({"kind": "nmclock_node", "node": global_node}))
    sources = copy_reference(run, source)
    assert len(check_imported_sources(run, source, sources)) == 4
    (run/"v6_g.txt").write_text("changed")
    with pytest.raises(ValueError, match="native artifact changed"):
        check_imported_sources(run, source, sources)


def test_selection_verification_handles_serialized_month_keys_without_hiding_changed_values():
    fields = ["config_sha256", "model_code_sha256", "selection_code_sha256", "paired_day_code_sha256",
              "bundle_id", "parent_bundle_id", "candidate_count", "selected_candidate", "refit_iterations"]
    rebuilt = {name: "same" for name in fields}
    rebuilt.update(candidates={"recipe": {"by_month": {1: {"rmse": 10.}}}},
                   reference_metrics={}, selected_paired_days={})
    saved = json.loads(json.dumps(rebuilt))
    compare_selection(saved, rebuilt)
    bad = deepcopy(saved)
    bad["candidates"]["recipe"]["by_month"]["1"]["rmse"] = 11.
    with pytest.raises(ValueError):
        compare_selection(bad, rebuilt)
