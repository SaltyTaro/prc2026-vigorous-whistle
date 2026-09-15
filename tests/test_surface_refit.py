"""Protect the final audit, declared counts, route weights and fallback values."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from prc2026.artifacts import sha256_file, write_json
from prc2026.features import ID, TIME
from prc2026_surface_airports.training import model_code_receipt
from prc2026_surface_refit import inference
from prc2026_surface_refit.guards import eligible_recipe, refit_assignment, require_verifications

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    config = json.loads((ROOT / "configs/surface_airports_v1.json").read_text())
    selection = json.loads((ROOT / "docs/surface_airports/development_results.json").read_text())
    timeline = json.loads((ROOT / "docs/timeline/development_results.json").read_text())
    return config, selection, timeline, {"bundle_id": selection["bundle_id"]}, model_code_receipt(ROOT)


def test_surface_refit_reserves_all_december_flight_groups():
    frame = pd.DataFrame({ID: [5, 2, 9, 1, 7], TIME: pd.to_datetime(["2025-11-30T23:00Z", "2025-12-01T01:00Z", "2025-10-05T01:00Z", "2025-12-20T01:00Z", "2025-01-01T01:00Z"], utc=True),
                          "FLIGHT_ID_mvt": ["crosses", "crosses", "ordinary", None, None]})
    assignment = refit_assignment(frame, {"reserved_audit_months": ["2025-12"]}, [f"2025-{m:02}" for m in range(1, 12)])
    assert assignment[ID].tolist() == [5, 2, 9, 1, 7]
    assert assignment.partition.tolist() == ["reserved", "reserved", "fit", "reserved", "fit"]


def test_surface_refit_rejects_december_refit_month():
    with pytest.raises(ValueError, match="January-November"):
        refit_assignment(pd.DataFrame(), {"reserved_audit_months": ["2025-12"]}, [f"2025-{m:02}" for m in range(1, 13)])


@pytest.mark.parametrize("change", ["median", "weights", "failed_fold", "incomplete"])
def test_surface_refit_requires_the_unchanged_eligible_selection(change):
    config, selection, timeline, manifest, code = inputs()
    eligible_recipe(config, selection, timeline, manifest, code)
    changed = deepcopy(selection)
    if change == "median":
        changed["refit_iterations"]["airports"]["LIRF"] += 1
    elif change == "weights":
        changed["candidates"][changed["selected_candidate"]]["routes"]["clock_available"]["cat_combined"] = .5
    elif change == "failed_fold":
        changed["candidates"][changed["selected_candidate"]]["folds"]["seasonal"]["metrics"]["overall"]["rmse_seconds"] = 325.
    else:
        changed["native_fit_count"] = 19
    with pytest.raises(ValueError):
        eligible_recipe(config, changed, timeline, manifest, code)


def test_surface_refit_routing_preserves_signed_values_and_fallback(monkeypatch):
    frame = pd.DataFrame({ID: [3, 7, 2, 8], "nmclock_available": [1., 0., 1., 0.], "nmclock_proxy_s": [-100., np.nan, 20000., np.nan]})
    pipeline = inference.make_pipeline({"globals": {"cat_combined": {"offset": -20.}, "lgb_combined": {"offset": 10.}}, "airports": {}}, {})
    def global_predict(run, node, part, threads):
        assert part.nmclock_available.eq(1).all()
        return part.nmclock_proxy_s.to_numpy() + node["offset"]
    def airport_predict(run, nodes, part, threads):
        assert part.nmclock_available.eq(1).all()
        return part.nmclock_proxy_s.to_numpy() + 30.
    def fallback(run, reference, part, threads):
        assert part.nmclock_available.eq(0).all()
        return part[ID].to_numpy(dtype=float) * -123.456789123
    monkeypatch.setattr(inference, "node_prediction", global_predict)
    monkeypatch.setattr(inference, "airport_prediction", airport_predict)
    monkeypatch.setattr(inference, "frozen_prediction", fallback)
    result = inference.pipeline_prediction(Path("."), pipeline, frame, 1)
    np.testing.assert_array_equal(result[[0, 2]], [-87.5, 20012.5])
    np.testing.assert_array_equal(result[[1, 3]], frame[ID].to_numpy()[[1, 3]] * -123.456789123)


def test_surface_refit_pipeline_rejects_changed_weights():
    pipeline = inference.make_pipeline({"globals": {"cat_combined": {}, "lgb_combined": {}}, "airports": {}}, {})
    pipeline["clock_available_weights"]["cat_combined"] = .5
    with pytest.raises(ValueError, match="unchanged"):
        inference.pipeline_prediction(Path("."), pipeline, pd.DataFrame(), 1)


@pytest.mark.parametrize("change", ["failed", "stale"])
def test_surface_refit_requires_matching_verification_reports(tmp_path, change):
    config, selection, _, _, code = inputs()
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)
    report = {"passed": True, "audit_used": False, "selection_sha256": sha256_file(selection_path),
              "model_code_sha256": code["sha256"], "bundle_id": selection["bundle_id"],
              "selected_candidate": selection["selected_candidate"], "refit_iterations": selection["refit_iterations"],
              "native_fit_count": 20, "candidate_count": 4}
    for name in ["verification.json", "local_verification.json"]:
        write_json(tmp_path / "docs/surface_airports" / name, report)
    require_verifications(tmp_path, selection_path, selection, code)
    changed = deepcopy(report)
    changed["passed" if change == "failed" else "selection_sha256"] = False if change == "failed" else "0" * 64
    write_json(tmp_path / "docs/surface_airports/local_verification.json", changed)
    with pytest.raises(ValueError, match="Matching"):
        require_verifications(tmp_path, selection_path, selection, code)
