"""Declared direct and residual NM-clock comparisons on both frozen folds."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TARGET
from prc2026.phase1 import emit
from prc2026.validation import evaluate, identity_hash, require_development_fold
from prc2026_models.experiments import model_environment, new_run, predict, receipts
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_weather.training import model_code_receipt as previous_model_code
from .bundle import load_bundle
from .encoding import contract, fit_encoder, transform
from .features import FEATURE_NAMES, POLICY, route_mask, target_offset
from .inference import component_pipeline, copy_file_tree, node_prediction, pipeline_prediction


def model_code_receipt(root):
    hashes = dict(previous_model_code(root)['files'])
    paths = [root/'src/prc2026_nmclock'/n for n in ['__init__.py', 'encoding.py', 'inference.py', 'training.py']]
    paths += [root/'tools/run_nmclock.py', root/'requirements-nmclock.txt']
    hashes.update({p.relative_to(root).as_posix(): sha256_file(p) for p in paths})
    return {'files': hashes, 'sha256': object_hash(hashes)}


def settings_for(config, spec, airport=False):
    result = {k: deepcopy(config[k]) for k in ['threads', 'seed', spec['kind']]}
    if airport:
        if spec['kind'] != 'lightgbm':
            raise ValueError('Only the declared LightGBM airport nodes are fitted.')
        result['lightgbm']['parameters'].update(config['airport_lightgbm']['parameters'])
        for key in ['rounds', 'early_stopping_rounds']:
            result['lightgbm'][key] = config['airport_lightgbm'][key]
    return result


def validate_config(config, protocol):
    if config['feature_policy'] != POLICY or config['validation_protocol'] != protocol['protocol_id']:
        raise ValueError('Feature/validation declaration mismatch.')
    if config['audit_used'] or config['target_filter'] != 'none' or config['prediction_postprocess'] != 'none':
        raise ValueError('No target filtering, clipping or audit selection is permitted.')
    if config['refit_months'] != [f'2025-{m:02}' for m in range(1, 12)]:
        raise ValueError('The December audit must remain excluded.')
    if config['development_folds'] != ['seasonal', 'forward'] or not 1 <= config['threads'] <= 16:
        raise ValueError('Both original folds and the authorized CPU allocation are required.')
    if set(config['variant_order']) != set(config['variants']) or len(config['variant_order']) != len(config['variants']):
        raise ValueError('Declared variants must be complete and unique.')
    for fold in config['development_folds']:
        require_development_fold(protocol, fold)
    for name, spec in config['variants'].items():
        contract(spec)
        if spec['route'] != 'clock_available' or spec['kind'] not in {'lightgbm', 'catboost'}:
            raise ValueError('Only declared clock routes and CPU models are supported.')
        anchor = spec.get('airport_anchor')
        if anchor:
            other = config['variants'][anchor]
            if (other['kind'] != 'lightgbm' or other['target'] != spec['target']
                    or config['variant_order'].index(anchor) >= config['variant_order'].index(name)):
                raise ValueError('Airport anchors must precede their matching target formulation.')
        elif spec['kind'] != 'lightgbm':
            raise ValueError('CatBoost needs a declared matching LightGBM airport anchor.')
    if config['catboost']['parameters']['task_type'] != 'CPU':
        raise ValueError('This campaign uses CPU CatBoost.')
    from prc2026_phase5.inference import validate_weights
    for weights in config['candidate_families'].values():
        validate_weights(list(weights.values()))
        if set(weights)-set(config['variants']):
            raise ValueError('Undeclared family member.')
    if any(not np.isfinite(w) or not 0 < w <= 1 for w in config['blend_weights_new_candidate']):
        raise ValueError('Invalid candidate weights.')
    if not 0 < config['minimum_improvement_fraction_each_fold'] < 1:
        raise ValueError('A positive two-fold promotion margin is required.')


def reference_selection(root, config):
    path = root/config['reference_selection']
    if sha256_file(path) != config['reference_selection_sha256']:
        raise ValueError('Frozen v5 selection changed.')
    selection = read_json(path)
    if selection['audit_used'] or selection['selected_candidate'] != 'cat_weather_v4_100':
        raise ValueError('Expected the frozen v5 selection.')
    selected = selection['candidates'][selection['selected_candidate']]
    if selected['new_weight'] != 1 or selected['family_weights'] != {'cat_weather': 1.0}:
        raise ValueError('Frozen v5 composition changed.')
    return selection


def reference_predictions(root, config, fold, features, previous_runs):
    selection = reference_selection(root, config)
    card = selection['development_runs']['cat_weather'][fold]
    source = previous_runs/card['run_id']
    if sha256_file(source/'run.json') != card['run_sha256']:
        raise ValueError('Frozen v5 development run changed.')
    run = read_json(source/'run.json')
    if run['model_code']['sha256'] != previous_model_code(root)['sha256']:
        raise ValueError('Frozen v5 model source changed.')
    verify_receipts(source, run['artifacts'])
    saved = pd.read_parquet(source/'predictions.parquet')
    if len(saved) != len(features) or identity_hash(saved[ID]) != identity_hash(features[ID]):
        raise ValueError('Frozen reference must cover the full scoring population.')
    aligned = features[[ID]].merge(saved, on=ID, how='left', validate='one_to_one', sort=False)
    if not np.isfinite(aligned['prediction']).all():
        raise ValueError('Reference predictions must be finite.')
    return aligned, {'run_id': source.name, 'run_sha256': card['run_sha256'],
                     'predictions_sha256': sha256_file(source/'predictions.parquet'),
                     'reference_selection_sha256': config['reference_selection_sha256']}


def checked_runs(root, output, config, manifest):
    code = model_code_receipt(root)
    result = {}
    for path in sorted(output.glob('*/run.json')):
        report = read_json(path)
        if report['stage'] != 'development':
            continue
        if (report['config_sha256'] != object_hash(config) or report['bundle_id'] != manifest['bundle_id']
                or report['model_code']['sha256'] != code['sha256'] or report['audit_used']):
            raise ValueError('Existing new run has different code, configuration or features.')
        key = (report['variant'], report['fold'])
        if key in result:
            raise ValueError('Duplicate completed development run.')
        verify_receipts(path.parent, report['artifacts'])
        result[key] = (path.parent, report)
    return result


def fit_node(run: Path, name: str, spec: dict, settings: dict, features, truth,
             stop=None, stop_truth=None, iterations: int | None = None) -> dict:
    if (stop is None) != (stop_truth is None) or (iterations is not None and stop is not None):
        raise ValueError("A fixed refit cannot inspect stopping data.")
    if iterations is not None and iterations < 1:
        raise ValueError("Positive fixed iteration count required.")
    if not route_mask(features).all() or (stop is not None and not route_mask(stop).all()):
        raise ValueError("Training nodes require the declared clock-available route.")
    kind = spec["kind"]
    paths = {"model_path": name+("_model.txt" if kind == "lightgbm" else "_model.cbm"),
             "encoder_path": name+"_encoder.json", "history_path": name+"_history.json", "importance_path": name+"_importance.json"}
    if any((run/p).exists() for p in paths.values()):
        raise ValueError("Fitted components cannot be overwritten.")
    with ResourceMonitor() as monitor:
        encoder = fit_encoder(features, spec)
        x = transform(features, encoder)
        y = aligned_truth(features, truth)[TARGET].to_numpy(dtype=float)-target_offset(features, spec["target"])
        xs = ys = None
        if stop is not None:
            xs = transform(stop, encoder)
            ys = aligned_truth(stop, stop_truth)[TARGET].to_numpy(dtype=float)-target_offset(stop, spec["target"])
        native = settings[kind]
        params = dict(native["parameters"])
        rounds = native["rounds"] if iterations is None else iterations
        emit("nmclock_training", model=kind, component=name, rows=len(x), predictors=len(encoder["feature_names"]), maximum_iterations=rounds)
        if kind == "lightgbm":
            import lightgbm as lgb
            params.update(num_threads=settings["threads"], seed=settings["seed"], feature_fraction_seed=settings["seed"],
                          bagging_seed=settings["seed"], data_random_seed=settings["seed"], metric="rmse")
            train = lgb.Dataset(x, label=y, categorical_feature=encoder["categorical_columns"])
            history, kwargs = {}, {}
            callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history)]
            if xs is not None:
                validation = lgb.Dataset(xs, label=ys, reference=train, categorical_feature=encoder["categorical_columns"])
                kwargs = {"valid_sets": [validation], "valid_names": ["early_stop"]}
                callbacks.append(lgb.early_stopping(native["early_stopping_rounds"], first_metric_only=True))
            model = lgb.train(params, train, num_boost_round=rounds, callbacks=callbacks, **kwargs)
            best = int(model.best_iteration or model.current_iteration())
            model.save_model(str(run/paths["model_path"]), num_iteration=best)
            importance = model.feature_importance(importance_type="gain")
        elif kind == "catboost":
            from catboost import CatBoostRegressor, Pool
            if params.get("task_type") != "CPU":
                raise ValueError("CPU CatBoost required.")
            params.update(iterations=rounds, random_seed=settings["seed"], thread_count=settings["threads"])
            model, kwargs = CatBoostRegressor(**params), {"verbose": 100}
            if xs is not None:
                kwargs.update(eval_set=Pool(xs, ys, cat_features=encoder["categorical_columns"]),
                              early_stopping_rounds=native["early_stopping_rounds"], use_best_model=True)
            model.fit(Pool(x, y, cat_features=encoder["categorical_columns"]), **kwargs)
            best, history = int(model.tree_count_), model.get_evals_result()
            model.save_model(str(run/paths["model_path"]), format="cbm")
            importance = model.get_feature_importance()
        else:
            raise ValueError("Unknown model kind.")
        if iterations is not None and best != iterations:
            raise ValueError("Refit differs from the declared fixed iterations.")
        write_json(run/paths["encoder_path"], encoder)
        write_json(run/paths["history_path"], history)
        write_json(run/paths["importance_path"], dict(zip(encoder["feature_names"], map(float, importance))))
        card = {"model_kind": kind, "schema": "nmclock", "spec": spec, "settings": settings, "parameters": params,
                "iterations": best, **paths, "fit_rows": len(features), "fit_ids_sha256": identity_hash(features[ID]),
                "early_stop_rows": 0 if stop is None else len(stop),
                "early_stop_ids_sha256": None if stop is None else identity_hash(stop[ID]), "target_filter": "none"}
        sample = (features if stop is None else stop).iloc[:1000]
        before = predict(model, transform(sample, encoder), kind, settings["threads"])+target_offset(sample, spec["target"])
        difference = float(np.max(np.abs(before-node_prediction(run, card, sample, settings["threads"]))))
        if difference > 1e-9:
            raise ValueError("Native nmclock model does not reproduce after reload.")
        card["max_model_reload_difference_seconds"] = difference
        if stop is not None:
            card["early_stop_rmse_seconds"] = float(np.sqrt(np.mean((predict(model, xs, kind, settings["threads"])-ys)**2)))
    card["resources"] = monitor.stats
    return card



def suite(root, base, config_path, output, fold, previous_runs, variants=None):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original['descriptor']['protocol'])
    feature_config = root/'configs/features_nmclock_v1.json'
    if sha256_file(feature_config) != config['feature_configuration_sha256'] or read_json(feature_config) != manifest['descriptor']['config']:
        raise ValueError('Model declaration and feature configuration differ.')
    require_development_fold(original['descriptor']['protocol'], fold)
    existing = checked_runs(root, output, config, manifest)
    chosen = config['variant_order'] if variants is None else variants
    if set(chosen)-set(config['variants']):
        raise ValueError('Undeclared variant requested.')
    assignment = pd.read_parquet(parent/'splits'/f'{fold}.parquet')
    frames, truths, masks, routed, routed_truth = {}, {}, {}, {}, {}
    for role in ['fit', 'early_stop', 'score']:
        frames[role], truth = load_partition(base, parent, catalog, assignment, role, FEATURE_NAMES)
        truths[role] = aligned_truth(frames[role], truth)
        if identity_hash(frames[role][ID]) != original['folds'][fold]['identity_sha256'][role]:
            raise ValueError('Protected outer fold identities changed.')
        masks[role] = route_mask(frames[role])
        routed[role] = frames[role].loc[masks[role]].reset_index(drop=True)
        routed_truth[role] = truths[role].loc[masks[role]].reset_index(drop=True)
    reference, reference_receipt = reference_predictions(root, config, fold, frames['score'], previous_runs)
    ref_metric = evaluate(truths['score'], reference, original['ranking_month_weights'] if fold == 'seasonal' else None)
    expected_rmse = reference_selection(root, config)['candidates']['cat_weather_v4_100']['folds'][fold]['metrics']['overall']['rmse_seconds']
    if not np.isclose(ref_metric['overall']['rmse_seconds'], expected_rmse, rtol=1e-12, atol=1e-9):
        raise ValueError('The frozen v5 reference score changed.')
    ref_dir = output/'references'
    ref_dir.mkdir(parents=True, exist_ok=True)
    path = ref_dir/f'{fold}_v5.parquet'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), reference, check_exact=True)
    else:
        reference.to_parquet(path, index=False, compression='zstd')
    reference_receipt.update(v5_sha256=sha256_file(path), metrics=ref_metric)
    write_json(ref_dir/f'{fold}.json', reference_receipt)
    for variant in chosen:
        if (variant, fold) in existing:
            emit('nmclock_run_reused', variant=variant, fold=fold)
            continue
        spec = config['variants'][variant]
        run = new_run(output, f'{fold}_{variant}')
        with ResourceMonitor() as monitor:
            global_node = fit_node(run, 'global', spec, settings_for(config, spec),
                routed['fit'], routed_truth['fit'], routed['early_stop'], routed_truth['early_stop'])
            anchor_receipt = None
            if spec.get('airport_anchor'):
                key = (spec['airport_anchor'], fold)
                if key not in existing:
                    raise ValueError('The declared airport anchor must finish first.')
                anchor_dir, anchor = existing[key]
                verify_receipts(anchor_dir, anchor['artifacts'])
                source_airports = read_json(anchor_dir/'nodes.json')['airports']
                airports = copy_file_tree(run, anchor_dir, source_airports, 'anchor_')
                anchor_receipt = {'variant': key[0], 'run_id': anchor_dir.name, 'run_sha256': sha256_file(anchor_dir/'run.json')}
            else:
                airports = {}
                for airport in config['airports']:
                    fitting = routed['fit']['airport'].eq(airport).to_numpy()
                    stopping = routed['early_stop']['airport'].eq(airport).to_numpy()
                    if not fitting.any() or not stopping.any():
                        raise ValueError('Each declared airport must have fitting and stopping records.')
                    airports[airport] = fit_node(run, 'airport_'+airport, spec, settings_for(config, spec, airport=True),
                        routed['fit'].loc[fitting].reset_index(drop=True), routed_truth['fit'].loc[fitting].reset_index(drop=True),
                        routed['early_stop'].loc[stopping].reset_index(drop=True), routed_truth['early_stop'].loc[stopping].reset_index(drop=True))
            pipeline = component_pipeline(global_node, airports)
            component = pipeline_prediction(run, pipeline, routed['score'], config['threads'])
            values = reference['prediction'].to_numpy().copy()
            values[masks['score']] = component
            predictions = pd.DataFrame({ID: frames['score'][ID], 'prediction': values})
            predictions.to_parquet(run/'predictions.parquet', index=False, compression='zstd')
            pd.DataFrame({ID: routed['score'][ID], 'prediction': component}).to_parquet(run/'component_predictions.parquet', index=False, compression='zstd')
            metrics = evaluate(truths['score'], predictions, original['ranking_month_weights'] if fold == 'seasonal' else None)
            nodes = {'global': global_node, 'airports': airports}
            for name, value in [('nodes', nodes), ('pipeline', pipeline), ('metrics', metrics), ('config', config), ('reference', reference_receipt)]:
                write_json(run/(name+'.json'), value)
        report = {'created_at_utc': utc_now(), 'run_id': run.name, 'stage': 'development', 'variant': variant, 'fold': fold,
            'model_kind': 'ensemble', 'bundle_id': manifest['bundle_id'], 'parent_bundle_id': original['bundle_id'],
            'feature_code': manifest['code'], 'model_code': model_code_receipt(root), 'config_sha256': object_hash(config),
            'component_iterations': {'global': global_node['iterations'], 'airports': {a: n['iterations'] for a,n in airports.items()}},
            'newly_fitted_models': 1 if anchor_receipt else 11, 'airport_anchor': anchor_receipt,
            'best_iterations': global_node['iterations'], 'audit_used': False, 'target_filter': 'none', 'prediction_postprocess': 'none',
            'max_model_reload_difference_seconds': max(n['max_model_reload_difference_seconds'] for n in [global_node,*airports.values()]),
            'environment': model_environment(root), 'resources': monitor.stats, 'artifacts': receipts(run)}
        for role in frames:
            report[role+'_rows'], report[role+'_ids_sha256'] = len(frames[role]), identity_hash(frames[role][ID])
            report['clock_'+role+'_rows'], report['clock_'+role+'_ids_sha256'] = len(routed[role]), identity_hash(routed[role][ID])
        write_json(run/'run.json', report)
        existing[(variant, fold)] = (run, report)
        emit('nmclock_run_complete', variant=variant, fold=fold, metrics=metrics['overall'], resources=monitor.stats)
        gc.collect()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/nmclock_v1.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/nmclock'))
    parser.add_argument('--previous-runs', type=Path, default=Path('runs/weather'))
    parser.add_argument('--fold', required=True, choices=['seasonal', 'forward'])
    parser.add_argument('--variant', action='append')
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/'data/nmclock/latest.json')['bundle'])
    suite(root, base, root/args.config, root/args.output, args.fold, root/args.previous_runs, args.variant)
