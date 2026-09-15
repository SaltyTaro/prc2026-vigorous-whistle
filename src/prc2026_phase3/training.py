"""Controlled LightGBM feature ablations on the two frozen development folds."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import (ResourceMonitor, object_hash, read_json, sha256_file,
                               utc_now, write_json)
from prc2026.features import ID, TIME
from prc2026.phase1 import emit
from prc2026.validation import evaluate, identity_hash, require_development_fold
from prc2026_models.experiments import (load_model, model_environment, new_run, predict,
                                       receipts, save_model)
from .bundle import load_bundle, load_partition
from .encoding import aligned_target, fit_encoder, transform
from .features import POLICY, feature_contract


def model_code_receipt(root: Path) -> dict:
    paths = [root / "src/prc2026_phase3" / name for name in ["__init__.py", "encoding.py", "training.py"]]
    paths += [*sorted((root / "src/prc2026_models").glob("*.py")), root / "tools/run_phase3.py",
              root / "requirements-phase2.txt", root / "requirements-phase3.txt"]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def validate_config(config: dict, protocol: dict) -> None:
    if config["feature_policy"] != POLICY or config["validation_protocol"] != protocol["protocol_id"]:
        raise ValueError("Phase-3 configuration does not match the feature/validation policy.")
    if config["target_filter"] != "none" or config["prediction_postprocess"] != "none":
        raise ValueError("Phase-3 ablations do not filter labels or alter predictions.")
    if not 1 <= config["threads"] <= 16:
        raise ValueError("Phase 3 is limited to the authorized 16-vCPU worker.")
    if config["development_folds"] != ["seasonal", "forward"]:
        raise ValueError("Both original development folds are required.")
    for fold in config["development_folds"]:
        require_development_fold(protocol, fold)
    if set(config["refit_months"]) & set(protocol["reserved_audit_months"]):
        raise ValueError("Refit cannot consume the reserved audit period.")
    if config["refit_months"] != [f"2025-{m:02}" for m in range(1, 12)]:
        raise ValueError("The refit population must remain January-November.")
    for families in config["variants"].values():
        feature_contract(families)


def fit_model(config: dict, encoder: dict, x: pd.DataFrame, y: np.ndarray,
              x_stop: pd.DataFrame | None = None, y_stop: np.ndarray | None = None,
              iterations: int | None = None):
    import lightgbm as lgb
    spec = config["lightgbm"]
    if (x_stop is None) != (y_stop is None) or (iterations is not None and x_stop is not None):
        raise ValueError("A fixed refit cannot use stopping data; stopping inputs must be paired.")
    rounds = spec["rounds"] if iterations is None else iterations
    if rounds < 1:
        raise ValueError("Training iterations must be positive.")
    params = dict(spec["parameters"])
    params.update(num_threads=config["threads"], seed=config["seed"],
                  feature_fraction_seed=config["seed"], bagging_seed=config["seed"], data_random_seed=config["seed"])
    categorical = encoder["categorical_columns"]
    train = lgb.Dataset(x, label=y, categorical_feature=categorical)
    history, kwargs = {}, {}
    callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history)]
    if x_stop is not None:
        stop = lgb.Dataset(x_stop, label=y_stop, reference=train, categorical_feature=categorical)
        kwargs.update(valid_sets=[stop], valid_names=["early_stop"])
        callbacks.append(lgb.early_stopping(spec["early_stopping_rounds"], first_metric_only=True))
    model = lgb.train(params, train, num_boost_round=rounds, callbacks=callbacks, **kwargs)
    return model, int(model.best_iteration or model.current_iteration()), history, params


def train(root: Path, base: Path, config_path: Path, variant: str, fold: str,
          output: Path, label: str) -> Path:
    config = read_json(config_path)
    code = model_code_receipt(root)
    manifest, parent, original, catalog = load_bundle(root, base)
    protocol = original["descriptor"]["protocol"]
    validate_config(config, protocol)
    require_development_fold(protocol, fold)
    families = config["variants"][variant]
    names, _ = feature_contract(families)
    run = new_run(output, f"{fold}_{variant}_lightgbm")
    with ResourceMonitor() as monitor:
        assignment = pd.read_parquet(parent / "splits" / f"{fold}.parquet")
        fit, truth = load_partition(base, parent, catalog, assignment, "fit", names)
        stop, stop_truth = load_partition(base, parent, catalog, assignment, "early_stop", names)
        encoder = fit_encoder(fit, families)
        x, y = transform(fit, encoder), aligned_target(fit, truth, families)
        x_stop, y_stop = transform(stop, encoder), aligned_target(stop, stop_truth, families)
        fit_ids, stop_ids = identity_hash(fit[ID]), identity_hash(stop[ID])
        for role, actual in [("fit", fit_ids), ("early_stop", stop_ids)]:
            if actual != original["folds"][fold]["identity_sha256"][role]:
                raise ValueError("Frozen fitting/stopping IDs changed.")
        fit_rows, stop_rows = len(x), len(x_stop)
        del fit, truth, stop, stop_truth
        gc.collect()
        emit("phase3_training", variant=variant, fold=fold, predictors=len(names), fit_rows=fit_rows, run=str(run))
        model, best, history, params = fit_model(config, encoder, x, y, x_stop, y_stop)
        stop_rmse = float(np.sqrt(np.mean((predict(model, x_stop, "lightgbm", config["threads"])-y_stop)**2)))
        model_path = save_model(model, "lightgbm", run, best)
        write_json(run / "encoder.json", encoder)
        write_json(run / "training_history.json", history)
        write_json(run / "feature_importance.json", dict(zip(names, map(float, model.feature_importance(importance_type="gain")))))
        del x, y, x_stop, y_stop
        gc.collect()
        score, truth = load_partition(base, parent, catalog, assignment, "score", names)
        encoded = transform(score, encoder)
        values = predict(model, encoded, "lightgbm", config["threads"])
        reloaded = predict(load_model(model_path, "lightgbm"), encoded.iloc[:1000], "lightgbm", config["threads"])
        reload_difference = float(np.max(np.abs(values[:len(reloaded)]-reloaded)))
        if reload_difference > 1e-9:
            raise ValueError("Persisted model prediction mismatch.")
        predictions = pd.DataFrame({ID: score[ID], "prediction": values})
        score_ids = identity_hash(predictions[ID])
        if score_ids != original["folds"][fold]["identity_sha256"]["score"]:
            raise ValueError("Frozen scoring population changed.")
        metrics = evaluate(truth, predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
        predictions.to_parquet(run / "predictions.parquet", index=False, compression="zstd")
        write_json(run / "metrics.json", metrics)
        write_json(run / "config.json", config)
    write_json(run / "run.json", {
        "created_at_utc": utc_now(), "run_id": run.name, "stage": "development", "label": label,
        "model_kind": "lightgbm", "variant": variant, "families": families, "feature_names": names,
        "fold": fold, "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
        "feature_code": manifest["code"], "model_code": code, "config_sha256": object_hash(config),
        "parameters": params, "best_iterations": best, "early_stop_rmse_seconds": stop_rmse,
        "fit_rows": fit_rows, "early_stop_rows": stop_rows, "score_rows": len(predictions),
        "fit_ids_sha256": fit_ids, "early_stop_ids_sha256": stop_ids, "score_ids_sha256": score_ids,
        "max_model_reload_difference_seconds": reload_difference, "audit_used": False,
        "target_filter": "none", "prediction_postprocess": "none",
        "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run),
    })
    emit("phase3_run_complete", variant=variant, fold=fold, best_iterations=best,
         metrics=metrics["overall"], resources=monitor.stats, run=str(run))
    return run


def refit(root: Path, base: Path, config_path: Path, selection_path: Path, output: Path, label: str) -> Path:
    config, selection = read_json(config_path), read_json(selection_path)
    code = model_code_receipt(root)
    if selection["config_sha256"] != object_hash(config) or selection["model_code_sha256"] != code["sha256"]:
        raise ValueError("Selected configuration/training code changed before refit.")
    variant, iterations = selection["selected_variant"], selection["refit_iterations"]
    if variant is None or iterations < 1 or selection["audit_used"]:
        raise ValueError("No eligible development recipe for refit.")
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    if selection["bundle_id"] != manifest["bundle_id"]:
        raise ValueError("Selected feature bundle changed before refit.")
    families = config["variants"][variant]
    names, _ = feature_contract(families)
    run = new_run(output, f"refit_{variant}_lightgbm")
    with ResourceMonitor() as monitor:
        months = catalog[TIME].dt.strftime("%Y-%m")
        assignment = catalog[[ID]].copy()
        assignment["partition"] = np.where(months.isin(config["refit_months"]), "fit", "reserved")
        reserved = original["descriptor"]["protocol"]["reserved_audit_months"]
        protected = set(catalog.loc[months.isin(reserved), "FLIGHT_ID_mvt"].dropna())
        assignment.loc[catalog["FLIGHT_ID_mvt"].isin(protected), "partition"] = "reserved"
        features, truth = load_partition(base, parent, catalog, assignment, "fit", names)
        encoder = fit_encoder(features, families)
        x, y = transform(features, encoder), aligned_target(features, truth, families)
        fit_rows, fit_ids = len(x), identity_hash(features[ID])
        del features, truth
        gc.collect()
        emit("phase3_refit_training", variant=variant, fit_rows=fit_rows, iterations=iterations, audit_used=False)
        model, actual, history, params = fit_model(config, encoder, x, y, iterations=iterations)
        if actual != iterations:
            raise ValueError("Refit tree count differs from the frozen recipe.")
        model_path = save_model(model, "lightgbm", run, actual)
        write_json(run / "encoder.json", encoder)
        write_json(run / "config.json", config)
        write_json(run / "selection.json", selection)
        write_json(run / "training_history.json", history)
        write_json(run / "feature_importance.json", dict(zip(names, map(float, model.feature_importance(importance_type="gain")))))
        del x, y
        gc.collect()
        ranking = pd.read_parquet(base / "features/ranking.parquet", columns=[ID, *names])
        encoded = transform(ranking, encoder)
        values = predict(model, encoded, "lightgbm", config["threads"])
        restored = predict(load_model(model_path, "lightgbm"), encoded.iloc[:1000], "lightgbm", config["threads"])
        reload_difference = float(np.max(np.abs(values[:len(restored)]-restored)))
        if reload_difference > 1e-9 or identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Refit persistence or ranking ID check failed.")
        predictions = pd.DataFrame({ID: ranking[ID], "prediction": values})
        predictions.to_parquet(run / "ranking_predictions.parquet", index=False, compression="zstd")
    write_json(run / "run.json", {
        "created_at_utc": utc_now(), "run_id": run.name, "stage": "refit", "label": label,
        "model_kind": "lightgbm", "variant": variant, "families": families, "feature_names": names,
        "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
        "feature_code": manifest["code"], "model_code": code, "config_sha256": object_hash(config),
        "parameters": params, "iterations": iterations, "fit_rows": fit_rows, "fit_ids_sha256": fit_ids,
        "refit_months": config["refit_months"], "audit_used": False,
        "max_model_reload_difference_seconds": reload_difference,
        "ranking_rows": len(predictions), "ranking_ids_sha256": identity_hash(predictions[ID]),
        "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run),
    })
    emit("phase3_refit_complete", variant=variant, run=str(run), ranking_rows=len(predictions), resources=monitor.stats)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase3"))
    parser.add_argument("--label", default="azure-arcw0")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--variant", required=True)
    training.add_argument("--fold", choices=["seasonal", "forward", "final_audit"], required=True)
    fitting = sub.add_parser("refit")
    fitting.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / (args.bundle or read_json(root / "data/phase3/latest.json")["bundle"])
    if args.command == "train":
        train(root, base, root / args.config, args.variant, args.fold, root / args.output, args.label)
    else:
        refit(root, base, root / args.config, root / args.selection, root / args.output, args.label)
