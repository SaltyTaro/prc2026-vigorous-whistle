"""Three frozen weather comparisons on the two original complete development folds."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.phase1 import emit
from prc2026.validation import evaluate, identity_hash, require_development_fold
from prc2026_models.experiments import model_environment, new_run, predict, receipts
from prc2026_phase3.bundle import load_partition
from prc2026_phase4.features import prediction_offset
from prc2026_phase5.inference import validate_weights
from prc2026_phase5r.baseline import aligned_truth
from prc2026_phase5r.training import model_code_receipt as previous_model_code, reference_predictions, reference_selection
from .bundle import load_bundle
from .encoding import contract, fit_encoder, transform
from .features import FEATURE_NAMES, POLICY
from .inference import copy_legacy_pipeline, node_prediction, pipeline_prediction


def model_code_receipt(root: Path) -> dict:
    hashes = dict(previous_model_code(root)["files"])
    paths = [root/"src/prc2026_weather"/n for n in ["__init__.py", "encoding.py", "inference.py", "training.py"]]
    paths += [root/"tools/run_weather.py", root/"requirements-weather.txt"]
    hashes.update({p.relative_to(root).as_posix(): sha256_file(p) for p in paths})
    return {"files": hashes, "sha256": object_hash(hashes)}


def validate_config(config: dict, protocol: dict):
    if config["feature_policy"] != POLICY or config["validation_protocol"] != protocol["protocol_id"]:
        raise ValueError("Weather/validation declaration mismatch.")
    if config["target_filter"] != "none" or config["prediction_postprocess"] != "none" or config["audit_used"]:
        raise ValueError("No target filtering, clipping or audit selection is permitted.")
    if config["refit_months"] != [f"2025-{m:02}" for m in range(1, 12)]:
        raise ValueError("Refit must preserve the reserved audit.")
    if config["development_folds"] != ["seasonal", "forward"] or not 1 <= config["threads"] <= 16:
        raise ValueError("Both original folds and the authorized single worker are required.")
    for fold in config["development_folds"]:
        require_development_fold(protocol, fold)
    if set(config["variant_order"]) != set(config["variants"]) or len(config["variant_order"]) != len(config["variants"]):
        raise ValueError("Declared variants must be complete and unique.")
    for spec in config["variants"].values():
        contract(spec)
        if spec["kind"] not in {"lightgbm", "catboost"} or spec["route"] != "all":
            raise ValueError("Only declared global CPU weather models are supported.")
    if config["catboost"]["parameters"]["task_type"] != "CPU":
        raise ValueError("This campaign uses the authorized CPU worker.")
    for weights in config["candidate_families"].values():
        validate_weights(list(weights.values()))
        if set(weights)-set(config["variants"]):
            raise ValueError("Undeclared candidate member.")
    if any(not np.isfinite(w) or not 0 < w <= 1 for w in config["blend_weights_new_candidate"]):
        raise ValueError("Invalid common blend weights.")
    if not 0 < config["minimum_improvement_fraction_each_fold"] < 1:
        raise ValueError("A positive two-fold improvement margin is required.")


def settings_for(config, spec):
    return {k: deepcopy(config[k]) for k in ["threads", "seed", spec["kind"]]}


def fit_node(run: Path, name: str, spec: dict, settings: dict, features, truth,
             stop=None, stop_truth=None, iterations: int | None = None) -> dict:
    if (stop is None) != (stop_truth is None) or (iterations is not None and stop is not None):
        raise ValueError("A fixed refit cannot inspect stopping data.")
    if iterations is not None and iterations < 1:
        raise ValueError("Positive fixed iteration count required.")
    kind = spec["kind"]
    paths = {"model_path": name+("_model.txt" if kind == "lightgbm" else "_model.cbm"),
             "encoder_path": name+"_encoder.json", "history_path": name+"_history.json", "importance_path": name+"_importance.json"}
    if any((run/p).exists() for p in paths.values()):
        raise ValueError("Fitted components cannot be overwritten.")
    with ResourceMonitor() as monitor:
        encoder = fit_encoder(features, spec)
        x = transform(features, encoder)
        y = aligned_truth(features, truth)[TARGET].to_numpy(dtype=float)-prediction_offset(features, "lirf_missing_schedule")
        xs = ys = None
        if stop is not None:
            xs = transform(stop, encoder)
            ys = aligned_truth(stop, stop_truth)[TARGET].to_numpy(dtype=float)-prediction_offset(stop, "lirf_missing_schedule")
        native = settings[kind]
        params = dict(native["parameters"])
        rounds = native["rounds"] if iterations is None else iterations
        emit("weather_training", model=kind, component=name, rows=len(x), predictors=len(encoder["feature_names"]), maximum_iterations=rounds)
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
        card = {"model_kind": kind, "schema": "weather", "spec": spec, "settings": settings, "parameters": params,
                "iterations": best, **paths, "fit_rows": len(features), "fit_ids_sha256": identity_hash(features[ID]),
                "early_stop_rows": 0 if stop is None else len(stop),
                "early_stop_ids_sha256": None if stop is None else identity_hash(stop[ID]), "target_filter": "none"}
        sample = (features if stop is None else stop).iloc[:1000]
        before = predict(model, transform(sample, encoder), kind, settings["threads"])+prediction_offset(sample, "lirf_missing_schedule")
        difference = float(np.max(np.abs(before-node_prediction(run, card, sample, settings["threads"]))))
        if difference > 1e-9:
            raise ValueError("Native weather model does not reproduce after reload.")
        card["max_model_reload_difference_seconds"] = difference
        if stop is not None:
            card["early_stop_rmse_seconds"] = float(np.sqrt(np.mean((predict(model, xs, kind, settings["threads"])-ys)**2)))
    card["resources"] = monitor.stats
    return card


def checked_runs(root: Path, output: Path, config: dict, manifest: dict) -> dict:
    code = model_code_receipt(root)
    result = {}
    for path in sorted(output.glob("*/run.json")):
        report = read_json(path)
        if report["stage"] != "development":
            continue
        if (report["config_sha256"] != object_hash(config) or report["bundle_id"] != manifest["bundle_id"]
                or report["model_code"]["sha256"] != code["sha256"] or report["audit_used"]):
            raise ValueError("Existing new run has different code/configuration/features.")
        key = (report["variant"], report["fold"])
        if key in result:
            raise ValueError("Duplicate completed development run.")
        verify_receipts(path.parent, report["artifacts"])
        result[key] = (path.parent, report)
    return result


def suite(root: Path, base: Path, config_path: Path, output: Path, fold: str,
          prior_runs: Path, followup_runs: Path, variants: list[str] | None = None):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    feature_config = root/"configs/features_weather_v1.json"
    if sha256_file(feature_config) != config["feature_configuration_sha256"] or read_json(feature_config) != manifest["descriptor"]["config"]:
        raise ValueError("Model declaration and feature configuration differ.")
    require_development_fold(original["descriptor"]["protocol"], fold)
    existing = checked_runs(root, output, config, manifest)
    chosen = config["variant_order"] if variants is None else variants
    if set(chosen)-set(config["variants"]):
        raise ValueError("Undeclared variant requested.")
    assignment = pd.read_parquet(parent/"splits"/f"{fold}.parquet")
    frames, truths = {}, {}
    for role in ["fit", "early_stop", "score"]:
        frames[role], truths[role] = load_partition(base, parent, catalog, assignment, role, FEATURE_NAMES)
        if identity_hash(frames[role][ID]) != original["folds"][fold]["identity_sha256"][role]:
            raise ValueError("Protected outer fold IDs changed.")
    reference, airports, source_receipts = reference_predictions(root, config, fold, frames["score"], prior_runs, followup_runs)
    reference_metrics = evaluate(truths["score"], reference, original["ranking_month_weights"] if fold == "seasonal" else None)
    selection = reference_selection(root, config)
    expected_rmse = selection["candidates"][selection["selected_candidate"]]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
    if not np.isclose(reference_metrics["overall"]["rmse_seconds"], expected_rmse, rtol=1e-12, atol=1e-9):
        raise ValueError("Frozen v4 scores do not reproduce.")
    ref_dir = output/"references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for label, frame in [("v4", reference), ("airports", airports)]:
        dest = ref_dir/f"{fold}_{label}.parquet"
        if dest.is_file():
            pd.testing.assert_frame_equal(pd.read_parquet(dest), frame, check_exact=True)
        else:
            frame.to_parquet(dest, index=False, compression="zstd")
    reference_receipt = {"sources": source_receipts, "reference_selection_sha256": config["reference_selection_sha256"],
                         "v4_sha256": sha256_file(ref_dir/f"{fold}_v4.parquet"),
                         "airports_sha256": sha256_file(ref_dir/f"{fold}_airports.parquet"), "metrics": reference_metrics}
    write_json(ref_dir/f"{fold}.json", reference_receipt)
    for variant in chosen:
        if (variant, fold) in existing:
            emit("weather_run_reused", variant=variant, fold=fold)
            continue
        spec = config["variants"][variant]
        run = new_run(output, f"{fold}_{variant}")
        with ResourceMonitor() as monitor:
            node = fit_node(run, "new", spec, settings_for(config, spec), frames["fit"], truths["fit"],
                            frames["early_stop"], truths["early_stop"])
            mask = np.ones(len(frames["score"]), dtype=bool)
            component = node_prediction(run, node, frames["score"].loc[mask].reset_index(drop=True), config["threads"])
            if spec["route"] == "all":
                values = .5*component+.5*airports["prediction"].to_numpy()
            else:
                values = reference["prediction"].to_numpy().copy()
                values[mask] = component
            predictions = pd.DataFrame({ID: frames["score"][ID], "prediction": values})
            predictions.to_parquet(run/"predictions.parquet", index=False, compression="zstd")
            pd.DataFrame({ID: frames["score"].loc[mask, ID], "prediction": component}).to_parquet(
                run/"component_predictions.parquet", index=False, compression="zstd")
            metrics = evaluate(truths["score"], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
            write_json(run/"node.json", node)
            write_json(run/"metrics.json", metrics)
            write_json(run/"config.json", config)
            write_json(run/"reference.json", reference_receipt)
        report = {"created_at_utc": utc_now(), "run_id": run.name, "stage": "development", "variant": variant, "fold": fold,
                  "model_kind": "ensemble", "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
                  "feature_code": manifest["code"], "model_code": model_code_receipt(root), "config_sha256": object_hash(config),
                  "best_iterations": node["iterations"], "audit_used": False, "target_filter": "none", "prediction_postprocess": "none",
                  "max_model_reload_difference_seconds": node["max_model_reload_difference_seconds"],
                  "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run)}
        for role in frames:
            report[role+"_rows"], report[role+"_ids_sha256"] = len(frames[role]), identity_hash(frames[role][ID])
        write_json(run/"run.json", report)
        emit("weather_run_complete", variant=variant, fold=fold, iterations=node["iterations"], metrics=metrics["overall"], resources=monitor.stats)
        gc.collect()



def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/weather_v1.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/weather'))
    parser.add_argument('--prior-runs', type=Path, default=Path('runs/phase5'))
    parser.add_argument('--followup-runs', type=Path, default=Path('runs/phase5_followup'))
    parser.add_argument('--fold', required=True, choices=['seasonal', 'forward'])
    parser.add_argument('--variant', action='append')
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/'data/weather/latest.json')['bundle'])
    suite(root, base, root/args.config, root/args.output, args.fold, root/args.prior_runs, root/args.followup_runs, args.variant)
