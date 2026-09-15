"""Refit a verified NM-clock recipe on January-November, preserving the v5 fallback."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TIME
from prc2026.phase1 import emit
from prc2026.validation import identity_hash
from prc2026_models.experiments import model_environment, new_run, receipts
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_nmclock.bundle import load_bundle
from prc2026_nmclock.features import FEATURE_NAMES, route_mask
from prc2026_nmclock.inference import component_pipeline, copy_v5_pipeline, pipeline_prediction
from prc2026_nmclock.training import checked_runs, fit_node, model_code_receipt, settings_for, validate_config


def eligible_recipe(config, selection, manifest, code):
    if (selection['audit_used'] or selection['selected_candidate'] is None
            or selection['config_sha256'] != object_hash(config) or selection['bundle_id'] != manifest['bundle_id']
            or selection['model_code_sha256'] != code['sha256']):
        raise ValueError('An unchanged eligible selection is required before refit.')
    candidate = selection['candidates'][selection['selected_candidate']]
    if not candidate['eligible'] or candidate['family_weights'] != config['candidate_families'][candidate['variant']]:
        raise ValueError('The selected family was not declared or eligible.')
    if candidate['new_weight'] not in config['blend_weights_new_candidate']:
        raise ValueError('Undeclared final ensemble weight.')
    if set(candidate['folds']) != set(config['development_folds']):
        raise ValueError('Both complete validation folds are required.')
    for fold, row in candidate['folds'].items():
        reference = selection['reference_metrics'][fold]['overall']['rmse_seconds']
        ratio = row['metrics']['overall']['rmse_seconds']/reference
        if not np.isfinite(ratio) or ratio > 1-config['minimum_improvement_fraction_each_fold']:
            raise ValueError('Candidate fails the declared two-fold improvement margin.')
    if set(selection['refit_iterations']) != set(candidate['family_weights']):
        raise ValueError('Refit iteration families differ from the selected recipe.')
    for variant in candidate['family_weights']:
        cards = [selection['development_runs'][variant][fold]['component_iterations'] for fold in config['development_folds']]
        expected = {'global': int(np.rint(np.median([c['global'] for c in cards]))),
                    'airports': {a: int(np.rint(np.median([c['airports'][a] for c in cards]))) for a in config['airports']}}
        if selection['refit_iterations'][variant] != expected or min([expected['global'], *expected['airports'].values()]) < 1:
            raise ValueError('Fixed native iterations differ from the October-selected medians.')
    return candidate


def refit(root, base, config_path, output, selection_path, previous_runs):
    config, selection = read_json(config_path), read_json(selection_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original['descriptor']['protocol'])
    code = model_code_receipt(root)
    candidate = eligible_recipe(config, selection, manifest, code)
    if selection['selection_code_sha256'] != sha256_file(root/'tools/select_nmclock.py'):
        raise ValueError('Selection code changed after choosing the recipe.')
    experiments = checked_runs(root, output, config, manifest)
    if set(experiments) != {(v, f) for v in config['variant_order'] for f in config['development_folds']}:
        raise ValueError('Every declared experiment must finish before refit.')
    for (variant, fold), (directory, report) in experiments.items():
        saved = selection['development_runs'][variant][fold]
        if saved['run_id'] != directory.name or saved['run_sha256'] != sha256_file(directory/'run.json'):
            raise ValueError('Selected development run identity changed.')
    source = previous_runs/config['reference_refit_run']
    if sha256_file(source/'run.json') != config['reference_refit_sha256']:
        raise ValueError('Frozen v5 refit changed.')
    verify_receipts(source, read_json(source/'run.json')['artifacts'])
    if any(read_json(p)['stage'] == 'refit' for p in output.glob('*/run.json')):
        raise ValueError('A completed NM clock refit already exists; preserve it.')
    run = new_run(output, 'refit_'+selection['selected_candidate'])
    with ResourceMonitor() as monitor:
        months = catalog[TIME].dt.strftime('%Y-%m')
        protected = set(catalog.loc[months.isin(original['descriptor']['protocol']['reserved_audit_months']), 'FLIGHT_ID_mvt'].dropna())
        assignment = catalog[[ID]].copy()
        assignment['partition'] = np.where(months.isin(config['refit_months']) & ~catalog['FLIGHT_ID_mvt'].isin(protected), 'fit', 'reserved')
        features, truth = load_partition(base, parent, catalog, assignment, 'fit', FEATURE_NAMES)
        truth = aligned_truth(features, truth)
        mask = route_mask(features)
        routed, routed_truth = features.loc[mask].reset_index(drop=True), truth.loc[mask].reset_index(drop=True)
        cards, members = {}, []
        for variant, weight in candidate['family_weights'].items():
            spec, iterations = config['variants'][variant], selection['refit_iterations'][variant]
            global_node = fit_node(run, variant+'_global', spec, settings_for(config, spec), routed, routed_truth,
                                   iterations=iterations['global'])
            airport_spec = config['variants'][spec.get('airport_anchor', variant)]
            airports = {}
            for airport in config['airports']:
                selected = routed['airport'].eq(airport).to_numpy()
                airports[airport] = fit_node(run, variant+'_airport_'+airport, airport_spec,
                    settings_for(config, airport_spec, airport=True), routed.loc[selected].reset_index(drop=True),
                    routed_truth.loc[selected].reset_index(drop=True), iterations=iterations['airports'][airport])
            cards[variant] = {'global': global_node, 'airports': airports}
            members.append({'weight': weight, 'pipeline': component_pipeline(global_node, airports)})
        previous = copy_v5_pipeline(run, source)
        pipeline = {'kind': 'nmclock_patch', 'weight': candidate['new_weight'],
                    'base': {'kind': 'v5', 'pipeline': previous}, 'replacement': {'kind': 'mixture', 'members': members}}
        for name, value in [('pipeline', pipeline), ('nodes', cards), ('selection', selection), ('config', config)]:
            write_json(run/(name+'.json'), value)
        ranking = pd.read_parquet(base/'features/ranking.parquet', columns=[ID, *FEATURE_NAMES])
        if identity_hash(ranking[ID]) != original['ranking_ids_sha256']:
            raise ValueError('Ranking population changed.')
        values = pipeline_prediction(run, pipeline, ranking, config['threads'])
        pd.DataFrame({ID: ranking[ID], 'prediction': values}).to_parquet(run/'ranking_predictions.parquet', index=False, compression='zstd')
        indexes = np.unique(np.r_[np.linspace(0, len(ranking)-1, 1000, dtype=int),
                                  np.flatnonzero(~route_mask(ranking)), np.flatnonzero(ranking['lirf_missing_record'].eq(1))])
        restored = pipeline_prediction(run, read_json(run/'pipeline.json'), ranking.iloc[indexes].reset_index(drop=True), config['threads'])
        delta = float(np.max(np.abs(values[indexes]-restored)))
        if delta > 1e-8:
            raise ValueError('Saved final NM pipeline does not reproduce ranking predictions.')
        original_predictions = pd.read_parquet(source/'ranking_predictions.parquet')
        expected_fallback = ranking[[ID]].merge(original_predictions, on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
        missing = ~route_mask(ranking)
        fallback_delta = float(np.max(np.abs(values[missing]-expected_fallback[missing]))) if missing.any() else 0.
        if fallback_delta > 1e-8:
            raise ValueError('The missing-clock fallback differs from frozen v5.')
    total_iterations = sum(c['global']+sum(c['airports'].values()) for c in selection['refit_iterations'].values())
    report = {'created_at_utc': utc_now(), 'run_id': run.name, 'stage': 'refit', 'model_kind': 'ensemble',
        'variant': selection['selected_candidate'], 'iterations': total_iterations, 'component_iterations': selection['refit_iterations'],
        'bundle_id': manifest['bundle_id'], 'parent_bundle_id': original['bundle_id'], 'feature_code': manifest['code'], 'model_code': code,
        'refit_code_sha256': sha256_file(Path(__file__)), 'config_sha256': object_hash(config), 'selection_sha256': sha256_file(selection_path),
        'fit_rows': len(features), 'fit_ids_sha256': identity_hash(features[ID]), 'clock_fit_rows': len(routed), 'clock_fit_ids_sha256': identity_hash(routed[ID]),
        'refit_months': config['refit_months'], 'audit_used': False, 'target_filter': 'none', 'prediction_postprocess': 'none',
        'ranking_rows': len(ranking), 'ranking_ids_sha256': identity_hash(ranking[ID]), 'max_model_reload_difference_seconds': delta,
        'fallback_ranking_rows': int(missing.sum()), 'max_fallback_difference_seconds': fallback_delta,
        'source_refit_sha256': config['reference_refit_sha256'], 'environment': model_environment(root),
        'resources': monitor.stats, 'artifacts': receipts(run)}
    write_json(run/'run.json', report)
    emit('nmclock_refit_complete', run=str(run), ranking_rows=len(ranking), resources=monitor.stats)
    return run


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/nmclock_v1.json'))
    parser.add_argument('--runs', type=Path, default=Path('runs/nmclock'))
    parser.add_argument('--selection', type=Path, default=Path('runs/nmclock/selection.json'))
    parser.add_argument('--previous-runs', type=Path, default=Path('runs/weather'))
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/'data/nmclock/latest.json')['bundle'])
    refit(root, base, root/args.config, root/args.runs, root/args.selection, root/args.previous_runs)


if __name__ == '__main__':
    main()
