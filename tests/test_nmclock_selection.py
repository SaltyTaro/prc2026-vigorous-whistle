"""The routed refit requires both complete scores and per-component frozen counts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
from prc2026.artifacts import object_hash

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('nmclock_refit_tool', root/'tools/refit_nmclock.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    config = json.loads((root/'configs/nmclock_v1.json').read_text())
    counts = {f: {'component_iterations': {'global': g, 'airports': {a: n for a in config['airports']}}}
              for f,g,n in [('seasonal', 11, 21), ('forward', 14, 22)]}
    selection = {'audit_used': False, 'selected_candidate': 'example', 'config_sha256': object_hash(config),
        'bundle_id': 'bundle', 'model_code_sha256': 'code',
        'refit_iterations': {'cat_residual': {'global': 12, 'airports': {a: 22 for a in config['airports']}}},
        'development_runs': {'cat_residual': counts},
        'reference_metrics': {f: {'overall': {'rmse_seconds': 300.}} for f in config['development_folds']},
        'candidates': {'example': {'eligible': True, 'variant': 'cat_residual', 'family_weights': {'cat_residual': 1.},
            'new_weight': .5, 'folds': {f: {'metrics': {'overall': {'rmse_seconds': 280.}}} for f in config['development_folds']}}}}
    return config, selection, {'bundle_id': 'bundle'}, {'sha256': 'code'}


def test_nm_refit_requires_actual_two_fold_margin_and_unchanged_identity():
    config, selection, bundle, code = fixture()
    assert module.eligible_recipe(config, selection, bundle, code)['new_weight'] == .5
    for value in [299., float('nan')]:
        changed = deepcopy(selection)
        changed['candidates']['example']['folds']['seasonal']['metrics']['overall']['rmse_seconds'] = value
        with pytest.raises(ValueError, match='two-fold'):
            module.eligible_recipe(config, changed, bundle, code)
    changed = deepcopy(selection)
    changed['model_code_sha256'] = 'other'
    with pytest.raises(ValueError, match='unchanged'):
        module.eligible_recipe(config, changed, bundle, code)


def test_nm_refit_freezes_each_global_and_airport_count_and_declared_weights():
    config, selection, bundle, code = fixture()
    for branch in ['global', 'airports']:
        changed = deepcopy(selection)
        if branch == 'global':
            changed['refit_iterations']['cat_residual']['global'] += 1
        else:
            changed['refit_iterations']['cat_residual']['airports']['LIRF'] += 1
        with pytest.raises(ValueError, match='medians'):
            module.eligible_recipe(config, changed, bundle, code)
    changed = deepcopy(selection)
    changed['candidates']['example']['new_weight'] = .33
    with pytest.raises(ValueError, match='Undeclared'):
        module.eligible_recipe(config, changed, bundle, code)
