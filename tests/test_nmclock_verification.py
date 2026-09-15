"""Saved JSON and recomputed month keys must compare without hiding changed values."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from verify_nmclock import check_selection


def test_nmclock_verifier_compares_recomputed_months_with_saved_json():
    rebuilt = {key: 'unchanged' for key in [
        'config_sha256', 'model_code_sha256', 'selection_code_sha256', 'paired_day_code_sha256',
        'bundle_id', 'selected_candidate', 'refit_iterations']}
    rebuilt['candidates'] = {'example': {'folds': {'seasonal': {'metrics': {
        'ranking_month_weights': {1: .4, 7: .6}, 'overall': {'rmse_seconds': 324.9}}}}}}
    saved = json.loads(json.dumps(rebuilt))
    check_selection(saved, rebuilt)
    changed = deepcopy(rebuilt)
    changed['candidates']['example']['folds']['seasonal']['metrics']['overall']['rmse_seconds'] = 300.
    with pytest.raises(ValueError, match='Metric differs'):
        check_selection(saved, changed)
    colliding = deepcopy(rebuilt)
    colliding['candidates']['example']['folds']['seasonal']['metrics']['ranking_month_weights']['1'] = .4
    with pytest.raises(ValueError, match='Metric keys differ'):
        check_selection(saved, colliding)
    changed = deepcopy(rebuilt)
    changed['selected_candidate'] = 'different'
    with pytest.raises(ValueError, match='Selection no longer reproduces'):
        check_selection(saved, changed)
