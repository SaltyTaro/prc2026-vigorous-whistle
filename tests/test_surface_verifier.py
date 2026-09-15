"""Recomputed integer month keys must compare safely with saved JSON keys."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_surface_airports import compare_saved_metrics


def test_surface_verifier_compares_recomputed_metrics_to_json_roundtrip():
    rebuilt = {"seasonal": {"ranking_month_weights": {1: .45, 7: .55}, "overall": {"rows": 344419, "rmse_seconds": 322.8193230817628}}}
    saved = json.loads(json.dumps(rebuilt))
    compare_saved_metrics(rebuilt=rebuilt, saved=saved)


def test_surface_verifier_still_rejects_changed_weights_and_key_collisions():
    rebuilt = {"ranking_month_weights": {1: .45, 7: .55}}
    saved = json.loads(json.dumps(rebuilt))
    changed = deepcopy(saved)
    changed["ranking_month_weights"]["7"] = .5
    with pytest.raises(ValueError):
        compare_saved_metrics(rebuilt=rebuilt, saved=changed)
    collision = {"ranking_month_weights": {1: .45, "1": .45, 7: .55}}
    with pytest.raises(ValueError):
        compare_saved_metrics(rebuilt=collision, saved=saved)
