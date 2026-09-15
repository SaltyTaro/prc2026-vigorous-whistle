"""Verify all clock features, protected populations, complete scores and native pipelines."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TIME
from prc2026.phase1 import validate_inputs
from prc2026.validation import evaluate, identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_weather.inference import node_prediction as weather_node, pipeline_prediction as weather_pipeline
from prc2026_nmclock.bundle import load_bundle
from prc2026_nmclock.features import (FEATURE_NAMES, INPUT_COLUMNS, PREVIOUS_NAMES, CLOCK_NAMES,
    clock_features, coverage, route_mask)
from prc2026_nmclock.inference import node_prediction, pipeline_prediction
from prc2026_nmclock.training import checked_runs, model_code_receipt, reference_predictions, settings_for, validate_config
from select_nmclock import collect
from refit_nmclock import eligible_recipe
from verify_phase5 import same_metrics


def difference(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Comparisons require matching finite arrays.')
    return float(np.max(np.abs(left-right))) if left.size else 0.


def sample(frame, include_missing=True):
    exceptional = frame['nmclock_proxy_s'].lt(0) | frame['nmclock_proxy_s'].gt(10800)
    if include_missing:
        exceptional |= ~pd.Series(route_mask(frame), index=frame.index)
    indexes = np.unique(np.r_[np.linspace(0, len(frame)-1, min(1000, len(frame)), dtype=int),
                              np.flatnonzero(exceptional)])
    return frame.iloc[indexes].reset_index(drop=True)


def check_card(card, spec, settings, fitting_ids, stopping_ids=None, fixed_iterations=None):
    if card['spec'] != spec or card['settings'] != settings or card['model_kind'] != spec['kind']:
        raise ValueError('Native model differs from its declared feature, target or settings.')
    if card['fit_rows'] != len(fitting_ids) or card['fit_ids_sha256'] != identity_hash(fitting_ids):
        raise ValueError('Native fitting population changed.')
    if card['target_filter'] != 'none':
        raise ValueError('Native targets were filtered.')
    if stopping_ids is None:
        if card['early_stop_rows'] != 0 or card['early_stop_ids_sha256'] is not None or card['iterations'] != fixed_iterations:
            raise ValueError('Refit used stopping data or different fixed iterations.')
    elif (card['early_stop_rows'] != len(stopping_ids) or card['early_stop_ids_sha256'] != identity_hash(stopping_ids)
          or not 1 <= card['iterations'] <= settings[spec['kind']]['rounds']):
        raise ValueError('Native stopping population or selected iteration count changed.')


def independent_component(run, nodes, frame, threads):
    global_values = node_prediction(run, nodes['global'], frame, threads)
    airports = np.full(len(frame), np.nan)
    for airport, node in nodes['airports'].items():
        chosen = frame['airport'].eq(airport).to_numpy()
        if chosen.any():
            airports[chosen] = node_prediction(run, node, frame.loc[chosen].reset_index(drop=True), threads)
    if not np.isfinite(airports).all():
        raise ValueError('Airport components do not cover the requested route.')
    return .5*global_values+.5*airports


def check_selection(saved, rebuilt):
    for key in ['config_sha256', 'model_code_sha256', 'selection_code_sha256', 'paired_day_code_sha256',
                'bundle_id', 'selected_candidate', 'refit_iterations']:
        if saved[key] != rebuilt[key]:
            raise ValueError('Selection no longer reproduces: '+key)
    # The helper accepts recomputed values first and the serialized receipt second.
    # Only the former can still contain integer month keys.
    same_metrics(rebuilt['candidates'], saved['candidates'])


def verify(root, base, config_path, output, previous_runs, require_refit=False):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original['descriptor']['protocol'])
    if validate_inputs(root) != manifest['descriptor']['input_files']:
        raise ValueError('Original competition inputs changed.')
    counts, feature_rows, cyclic_max = {}, 0, 0.
    for source, rows in manifest['rows'].items():
        frame = pd.read_parquet(base/'features'/source)
        previous = pd.read_parquet(root/manifest['previous_bundle']/'features'/source)
        if list(frame) != [ID, *FEATURE_NAMES] or len(frame) != rows:
            raise ValueError('New feature schema or population changed.')
        pd.testing.assert_frame_equal(frame[[ID, *PREVIOUS_NAMES]], previous, check_exact=True)
        if identity_hash(frame[ID]) != manifest['source_ids_sha256'][source]:
            raise ValueError('New feature identity changed.')
        raw = pd.read_parquet(root/'data/raw'/source, columns=INPUT_COLUMNS)
        query = frame[[ID]].merge(raw, on=ID, how='left', validate='one_to_one', sort=False)
        rebuilt = clock_features(query)
        cyclic = [n for n in CLOCK_NAMES if n.endswith(('_utc_hour_sin', '_utc_hour_cos'))]
        exact = [ID, *[n for n in CLOCK_NAMES if n not in cyclic]]
        pd.testing.assert_frame_equal(rebuilt[exact], frame[exact], check_exact=True)
        a, b = rebuilt[cyclic].to_numpy(), frame[cyclic].to_numpy()
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-14, equal_nan=True)
        np.testing.assert_array_equal(a.astype('float32'), b.astype('float32'))
        finite = np.abs(a-b)[np.isfinite(a-b)]
        cyclic_max = max(cyclic_max, float(finite.max()) if finite.size else 0.)
        # Separate arithmetic check, without the feature helper's proxy selection.
        chosen_clock = query['AOBT_3_flt'].where(query['AOBT_3_flt'].notna(), query['LOBT_flt'])
        expected_proxy = (query[TIME]-chosen_clock).dt.total_seconds().to_numpy(dtype=float)
        np.testing.assert_allclose(frame['nmclock_proxy_s'], expected_proxy, rtol=0, atol=0, equal_nan=True)
        counts[source] = coverage(frame)
        feature_rows += len(frame)
    same_metrics(counts, manifest['coverage'])
    experiments = checked_runs(root, output, config, manifest)
    if set(experiments) != {(v, f) for v in config['variant_order'] for f in config['development_folds']}:
        raise ValueError('The entire declared batch must finish before verification.')
    native, run_hashes, scores, reference_checks = {}, {}, {}, {}
    for fold in config['development_folds']:
        assignment = pd.read_parquet(parent/'splits'/f'{fold}.parquet')
        features, truth = load_partition(base, parent, catalog, assignment, 'score', FEATURE_NAMES)
        reference, source_receipt = reference_predictions(root, config, fold, features, previous_runs)
        pd.testing.assert_frame_equal(reference, pd.read_parquet(output/'references'/f'{fold}_v5.parquet'), check_exact=True)
        source = previous_runs/source_receipt['run_id']
        chosen_reference = sample(features)
        airport_file = previous_runs/'references'/f'{fold}_airports.parquet'
        if sha256_file(airport_file) != read_json(source/'reference.json')['airports_sha256']:
            raise ValueError('Frozen v5 airport reference changed.')
        airports = chosen_reference[[ID]].merge(pd.read_parquet(airport_file), on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
        actual = .5*weather_node(source, read_json(source/'node.json'), chosen_reference, config['threads'])+.5*airports
        expected = chosen_reference[[ID]].merge(reference, on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
        delta = difference(actual, expected)
        if delta > 1e-8:
            raise ValueError('The original native v5 model no longer reproduces.')
        reference_checks[fold] = {'rows': len(chosen_reference), 'max_difference_seconds': delta}
        routed = features.loc[route_mask(features)].reset_index(drop=True)
        chosen = sample(routed, include_missing=False)
        role_features = {}
        for role in ['fit', 'early_stop']:
            minimum, _ = load_partition(base, parent, catalog, assignment, role, ['airport', 'nmclock_available', 'nmclock_proxy_s'])
            role_features[role] = minimum.loc[route_mask(minimum)].reset_index(drop=True)
        for variant in config['variant_order']:
            directory, run = experiments[(variant, fold)]
            spec = config['variants'][variant]
            nodes = read_json(directory/'nodes.json')
            for role in ['fit', 'early_stop', 'score']:
                if run[role+'_ids_sha256'] != original['folds'][fold]['identity_sha256'][role]:
                    raise ValueError('Protected full partition identities changed.')
                route_frame = routed if role == 'score' else role_features[role]
                if run['clock_'+role+'_rows'] != len(route_frame) or run['clock_'+role+'_ids_sha256'] != identity_hash(route_frame[ID]):
                    raise ValueError('Routed partition identity changed.')
            check_card(nodes['global'], spec, settings_for(config, spec), role_features['fit'][ID], role_features['early_stop'][ID])
            airport_spec = config['variants'][spec.get('airport_anchor', variant)]
            if set(nodes['airports']) != set(config['airports']):
                raise ValueError('Airport models are missing or undeclared.')
            for airport, node in nodes['airports'].items():
                fit_ids = role_features['fit'].loc[role_features['fit']['airport'].eq(airport), ID]
                stop_ids = role_features['early_stop'].loc[role_features['early_stop']['airport'].eq(airport), ID]
                check_card(node, airport_spec, settings_for(config, airport_spec, airport=True), fit_ids, stop_ids)
            counts_from_nodes = {'global': nodes['global']['iterations'], 'airports': {a: n['iterations'] for a,n in nodes['airports'].items()}}
            if run['component_iterations'] != counts_from_nodes:
                raise ValueError('Saved native iteration counts changed.')
            if spec.get('airport_anchor'):
                anchor_dir, _ = experiments[(spec['airport_anchor'], fold)]
                if run['airport_anchor']['run_sha256'] != sha256_file(anchor_dir/'run.json'):
                    raise ValueError('Native airport anchor changed.')
                anchor_nodes = read_json(anchor_dir/'nodes.json')['airports']
                for airport, node in nodes['airports'].items():
                    for key in ['model_path', 'encoder_path', 'history_path', 'importance_path']:
                        if sha256_file(directory/node[key]) != sha256_file(anchor_dir/anchor_nodes[airport][key]):
                            raise ValueError('Copied airport model differs from its declared anchor.')
            component = pd.read_parquet(directory/'component_predictions.parquet')
            if len(component) != len(routed) or identity_hash(component[ID]) != identity_hash(routed[ID]):
                raise ValueError('Native predictions must cover the complete clock route.')
            expected = chosen[[ID]].merge(component, on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
            delta = difference(independent_component(directory, nodes, chosen, config['threads']), expected)
            pipeline_delta = difference(pipeline_prediction(directory, read_json(directory/'pipeline.json'), chosen, config['threads']), expected)
            if max(delta, pipeline_delta) > 1e-8:
                raise ValueError('Native clock composition does not reproduce.')
            native[directory.name] = {'rows': len(chosen), 'max_difference_seconds': max(delta, pipeline_delta)}
            values = reference['prediction'].to_numpy().copy()
            values[route_mask(features)] = routed[[ID]].merge(component, on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
            saved = pd.read_parquet(directory/'predictions.parquet')
            expected = features[[ID]].merge(saved, on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
            if len(saved) != len(features) or difference(values, expected) > 1e-8:
                raise ValueError('Full scoring population or v5 fallback changed.')
            metrics = evaluate(truth, saved, original['ranking_month_weights'] if fold == 'seasonal' else None)
            same_metrics(metrics, read_json(directory/'metrics.json'))
            scores[directory.name], run_hashes[directory.name] = metrics['overall']['rmse_seconds'], sha256_file(directory/'run.json')
    selection, rebuilt = read_json(output/'selection.json'), collect(root, base, config_path, output)
    check_selection(selection, rebuilt)
    refits = verify_refits(root, base, output, previous_runs, config, selection, manifest, parent, original, catalog)
    if require_refit and not refits:
        raise ValueError('A verified refit was required but is missing.')
    return {'created_at_utc': utc_now(), 'passed': True, 'audit_used': False, 'bundle_id': manifest['bundle_id'],
        'model_code_sha256': model_code_receipt(root)['sha256'], 'verification_code_sha256': sha256_file(Path(__file__)),
        'feature_rows_checked': feature_rows, 'clock_feature_rows_recomputed': feature_rows, 'coverage': counts,
        'maximum_cyclic_float64_reconstruction_difference': cyclic_max, 'native_float32_cyclic_features_exact': True,
        'reference_native_checks': reference_checks, 'native_checks': native, 'development_scores': scores,
        'development_run_sha256': run_hashes, 'blend_fold_scores_checked': 2*len(selection['candidates']),
        'selection_sha256': sha256_file(output/'selection.json'), 'selected_candidate': selection['selected_candidate'], 'refits': refits}


def verify_refits(root, base, output, previous_runs, config, selection, manifest, parent, original, catalog):
    result = []
    for filename in sorted(output.glob('*/run.json')):
        report = read_json(filename)
        if report['stage'] != 'refit':
            continue
        run = filename.parent
        verify_receipts(run, report['artifacts'])
        if (report['audit_used'] or report['refit_months'] != config['refit_months']
                or report['config_sha256'] != object_hash(config) or report['bundle_id'] != manifest['bundle_id']
                or report['model_code']['sha256'] != model_code_receipt(root)['sha256']
                or report['selection_sha256'] != sha256_file(output/'selection.json')
                or report['refit_code_sha256'] != sha256_file(root/'tools/refit_nmclock.py')):
            raise ValueError('Refit provenance changed.')
        candidate = eligible_recipe(config, selection, manifest, model_code_receipt(root))
        if report['variant'] != selection['selected_candidate']:
            raise ValueError('Refit differs from the selected recipe.')
        months = catalog[TIME].dt.strftime('%Y-%m')
        protected = set(catalog.loc[months.isin(original['descriptor']['protocol']['reserved_audit_months']), 'FLIGHT_ID_mvt'].dropna())
        mask = months.isin(config['refit_months']) & ~catalog['FLIGHT_ID_mvt'].isin(protected)
        ids = catalog.loc[mask, ID]
        if report['fit_rows'] != len(ids) or report['fit_ids_sha256'] != identity_hash(ids):
            raise ValueError('Refit consumed different rows or protected flight groups.')
        assignment = catalog[[ID]].copy()
        assignment['partition'] = np.where(mask, 'fit', 'reserved')
        minimum, _ = load_partition(base, parent, catalog, assignment, 'fit', ['airport', 'nmclock_available', 'nmclock_proxy_s'])
        routed_fit = minimum.loc[route_mask(minimum)]
        if report['clock_fit_rows'] != len(routed_fit) or report['clock_fit_ids_sha256'] != identity_hash(routed_fit[ID]):
            raise ValueError('Refit clock route changed.')
        ranking = pd.read_parquet(base/'features/ranking.parquet')
        saved = pd.read_parquet(run/'ranking_predictions.parquet')
        if len(saved) != len(ranking) or identity_hash(saved[ID]) != original['ranking_ids_sha256']:
            raise ValueError('Ranking predictions must retain the complete template population.')
        chosen = sample(ranking)
        expected = chosen[[ID]].merge(saved, on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
        delta = difference(pipeline_prediction(run, read_json(run/'pipeline.json'), chosen, config['threads']), expected)
        if delta > 1e-8:
            raise ValueError('Final native NM pipeline does not reproduce.')
        cards = read_json(run/'nodes.json')
        if set(cards) != set(candidate['family_weights']):
            raise ValueError('Refit has missing or undeclared native families.')
        active = route_mask(chosen)
        proposed = np.zeros(int(active.sum()))
        for variant, weight in candidate['family_weights'].items():
            spec, iterations = config['variants'][variant], selection['refit_iterations'][variant]
            nodes = cards[variant]
            check_card(nodes['global'], spec, settings_for(config, spec), routed_fit[ID], fixed_iterations=iterations['global'])
            airport_spec = config['variants'][spec.get('airport_anchor', variant)]
            if set(nodes['airports']) != set(config['airports']):
                raise ValueError('Refit airport components changed.')
            for airport, node in nodes['airports'].items():
                airport_ids = routed_fit.loc[routed_fit['airport'].eq(airport), ID]
                check_card(node, airport_spec, settings_for(config, airport_spec, airport=True),
                           airport_ids, fixed_iterations=iterations['airports'][airport])
            proposed += weight*independent_component(run, nodes, chosen.loc[active].reset_index(drop=True), config['threads'])
        source = previous_runs/config['reference_refit_run']
        if sha256_file(source/'run.json') != config['reference_refit_sha256']:
            raise ValueError('Original v5 refit changed.')
        verify_receipts(source, read_json(source/'run.json')['artifacts'])
        old_values = weather_pipeline(source, read_json(source/'pipeline.json'), chosen, config['threads'])
        composed = old_values.copy()
        weight = candidate['new_weight']
        composed[active] = weight*proposed+(1-weight)*old_values[active]
        if difference(composed, expected) > 1e-8:
            raise ValueError('Final pipeline does not implement the declared composition.')
        original_predictions = pd.read_parquet(source/'ranking_predictions.parquet')
        old_values = ranking[[ID]].merge(original_predictions, on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
        new_values = ranking[[ID]].merge(saved, on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
        missing = ~route_mask(ranking)
        fallback_delta = difference(new_values[missing], old_values[missing])
        if fallback_delta > 1e-8 or report['fallback_ranking_rows'] != int(missing.sum()):
            raise ValueError('Complete missing-clock fallback changed.')
        result.append({'run_id': run.name, 'run_sha256': sha256_file(filename), 'rows': len(chosen),
                       'max_difference_seconds': delta, 'fallback_rows': int(missing.sum()),
                       'max_fallback_difference_seconds': fallback_delta})
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/nmclock_v1.json'))
    parser.add_argument('--runs', type=Path, default=Path('runs/nmclock'))
    parser.add_argument('--previous-runs', type=Path, default=Path('runs/weather'))
    parser.add_argument('--output', type=Path, default=Path('docs/nmclock/verification.json'))
    parser.add_argument('--require-refit', action='store_true')
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/'data/nmclock/latest.json')['bundle'])
    with ResourceMonitor() as monitor:
        report = verify(root, base, root/args.config, root/args.runs, root/args.previous_runs, args.require_refit)
    report['resources'] = monitor.stats
    write_json(root/args.output, report)
    print({k: report[k] for k in ['passed', 'feature_rows_checked', 'clock_feature_rows_recomputed', 'blend_fold_scores_checked', 'selected_candidate']})


if __name__ == '__main__':
    main()
