"""Logged CPU baselines, frozen-fold evaluation, and audit-preserving refits."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import (ResourceMonitor, code_receipt, environment, object_hash,
                               read_json, sha256_file, utc_now, verify_receipts, write_json)
from prc2026.features import CATEGORY_SOURCES, FEATURE_NAMES, ID, POLICY, TARGET, TIME
from prc2026.phase1 import emit, load_partition
from prc2026.validation import evaluate, identity_hash, require_development_fold, validate_protocol
from .encoding import aligned_target, fit_encoder, transform


def model_code_receipt(root: Path) -> dict:
    paths = [*sorted((root / "src/prc2026_models").glob("*.py")),
             root / "tools/run_phase2.py", root / "requirements-phase2.txt"]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def model_environment(root: Path) -> dict:
    report = environment(root)
    report["packages"].update({name: importlib.metadata.version(name)
                               for name in ["lightgbm", "catboost", "scipy"]})
    return report


def validate_config(config: dict, protocol: dict) -> None:
    if config["feature_policy"] != POLICY or config["validation_protocol"] != protocol["protocol_id"]:
        raise ValueError("Baseline configuration does not match the frozen protocol.")
    if config["target_filter"] != "none" or config["prediction_postprocess"] != "none":
        raise ValueError("Baseline runner does not implement target filters or prediction transforms.")
    if not 1 <= config["threads"] <= 16:
        raise ValueError("Phase 2 is limited to one 16-vCPU worker.")
    for fold in config["development_folds"]:
        require_development_fold(protocol, fold)
    if set(config["refit_months"]) & set(protocol["reserved_audit_months"]):
        raise ValueError("Refitting cannot consume the reserved audit period.")


def load_bundle(root: Path, base: Path, config: dict):
    manifest = read_json(base / "manifest.json")
    protocol = manifest["descriptor"]["protocol"]
    validate_protocol(protocol)
    validate_config(config, protocol)
    if code_receipt(root)["sha256"] != manifest["code"]["sha256"]:
        raise ValueError("Frozen feature code changed; rebuild and review the feature bundle.")
    verify_receipts(base, manifest["artifacts"])
    return manifest, pd.read_parquet(base / "train_catalog.parquet")


def fit_model(kind: str, config: dict, x: pd.DataFrame, y: np.ndarray,
              x_stop: pd.DataFrame | None = None, y_stop: np.ndarray | None = None,
              fixed_iterations: int | None = None):
    spec = config[kind]
    if fixed_iterations is not None and fixed_iterations < 1:
        raise ValueError("Training iterations must be positive.")
    iterations = fixed_iterations or spec["rounds"]
    if iterations < 1:
        raise ValueError("Training iterations must be positive.")
    if (x_stop is None) != (y_stop is None):
        raise ValueError("Stopping features and labels must be supplied together.")
    if fixed_iterations is not None and x_stop is not None:
        raise ValueError("A fixed refit must not use early-stopping data.")
    params = dict(spec["parameters"])
    categories = list(CATEGORY_SOURCES)
    if kind == "lightgbm":
        import lightgbm as lgb
        params.update({"num_threads": config["threads"], "seed": config["seed"],
                       "feature_fraction_seed": config["seed"], "bagging_seed": config["seed"],
                       "data_random_seed": config["seed"]})
        train = lgb.Dataset(x, label=y, categorical_feature=categories)
        history = {}
        callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history)]
        kwargs = {}
        if x_stop is not None:
            stop = lgb.Dataset(x_stop, label=y_stop, reference=train, categorical_feature=categories)
            kwargs.update(valid_sets=[stop], valid_names=["early_stop"])
            callbacks.append(lgb.early_stopping(spec["early_stopping_rounds"], first_metric_only=True))
        model = lgb.train(params, train, num_boost_round=iterations, callbacks=callbacks, **kwargs)
        best = int(model.best_iteration or model.current_iteration())
        return model, best, history, params
    if kind == "catboost":
        from catboost import CatBoostRegressor, Pool
        params.update({"iterations": iterations, "thread_count": config["threads"],
                       "random_seed": config["seed"]})
        model = CatBoostRegressor(**params)
        train = Pool(x, y, cat_features=categories)
        kwargs = {"verbose": 100}
        if x_stop is not None:
            kwargs.update(eval_set=Pool(x_stop, y_stop, cat_features=categories),
                          early_stopping_rounds=spec["early_stopping_rounds"], use_best_model=True)
        model.fit(train, **kwargs)
        return model, int(model.tree_count_), model.get_evals_result(), params
    raise ValueError("Unknown model.")


def save_model(model, kind: str, run: Path, iterations: int) -> Path:
    path = run / ("model.txt" if kind == "lightgbm" else "model.cbm")
    if kind == "lightgbm":
        model.save_model(str(path), num_iteration=iterations)
    else:
        model.save_model(str(path), format="cbm")
    return path


def load_model(path: Path, kind: str):
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(str(path), format="cbm")
    return model


def predict(model, x: pd.DataFrame, kind: str, threads: int) -> np.ndarray:
    values = (model.predict(x, num_threads=threads) if kind == "lightgbm"
              else model.predict(x, thread_count=threads))
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(x),) or not np.isfinite(values).all():
        raise ValueError("Model did not produce one finite prediction per row.")
    return values


def receipts(run: Path) -> list:
    return [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(run.iterdir()) if p.is_file() and p.name != "run.json"]


def new_run(output: Path, name: str) -> Path:
    run = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + name)
    run.mkdir(parents=True, exist_ok=False)
    return run


def training_run(root: Path, base: Path, config_path: Path, kind: str, fold: str,
                 output: Path, label: str) -> Path:
    config = read_json(config_path)
    source_receipt = model_code_receipt(root)
    run = new_run(output, f"{fold}_{kind}")
    with ResourceMonitor() as monitor:
        manifest, catalog = load_bundle(root, base, config)
        protocol = manifest["descriptor"]["protocol"]
        require_development_fold(protocol, fold)
        if fold not in config["development_folds"]:
            raise ValueError("Fold not declared for baseline development.")
        assignment = pd.read_parquet(base / "splits" / f"{fold}.parquet")
        fit_features, fit_truth = load_partition(base, catalog, assignment, "fit")
        stop_features, stop_truth = load_partition(base, catalog, assignment, "early_stop")
        encoder = fit_encoder(fit_features, kind)
        x, y = transform(fit_features, encoder), aligned_target(fit_features, fit_truth)
        x_stop, y_stop = transform(stop_features, encoder), aligned_target(stop_features, stop_truth)
        fit_ids, stop_ids = identity_hash(fit_features[ID]), identity_hash(stop_features[ID])
        if fit_ids != manifest["folds"][fold]["identity_sha256"]["fit"]:
            raise ValueError("Fitting population differs from the frozen fold.")
        if stop_ids != manifest["folds"][fold]["identity_sha256"]["early_stop"]:
            raise ValueError("Early-stopping population differs from the frozen fold.")
        emit("baseline_training", model=kind, fold=fold, fit_rows=len(x), early_stop_rows=len(x_stop),
             maximum_iterations=config[kind]["rounds"], run=str(run))
        del fit_features, fit_truth, stop_features, stop_truth
        gc.collect()
        model, best, history, params = fit_model(kind, config, x, y, x_stop, y_stop)
        fit_rows, stop_rows = len(x), len(x_stop)
        stop_rmse = float(np.sqrt(np.mean((predict(model, x_stop, kind, config["threads"]) - y_stop) ** 2)))
        model_path = save_model(model, kind, run, best)
        write_json(run / "encoder.json", encoder)
        write_json(run / "training_history.json", history)
        importance = (model.feature_importance(importance_type="gain") if kind == "lightgbm"
                      else model.get_feature_importance())
        write_json(run / "feature_importance.json",
                   {name: float(value) for name, value in zip(FEATURE_NAMES, importance)})
        del x, y, x_stop, y_stop
        gc.collect()
        score_features, truth = load_partition(base, catalog, assignment, "score")
        encoded = transform(score_features, encoder)
        values = predict(model, encoded, kind, config["threads"])
        reloaded = load_model(model_path, kind)
        restored = predict(reloaded, encoded.iloc[:1000], kind, config["threads"])
        max_reload_difference = float(np.max(np.abs(values[:len(restored)] - restored)))
        if max_reload_difference > 1e-9:
            raise ValueError("Persisted model does not reproduce predictions.")
        predictions = pd.DataFrame({ID: score_features[ID], "prediction": values})
        if identity_hash(predictions[ID]) != manifest["folds"][fold]["identity_sha256"]["score"]:
            raise ValueError("Validation scoring population changed.")
        metrics = evaluate(truth, predictions,
                           manifest["ranking_month_weights"] if fold == "seasonal" else None)
        predictions.to_parquet(run / "predictions.parquet", index=False, compression="zstd")
        write_json(run / "metrics.json", metrics)
        write_json(run / "config.json", config)
    report = {
        "run_id": run.name, "created_at_utc": utc_now(), "stage": "development", "label": label,
        "model_kind": kind, "fold": fold, "bundle_id": manifest["bundle_id"],
        "feature_code": manifest["code"], "model_code": source_receipt,
        "config_sha256": object_hash(config), "parameters": params, "best_iterations": best,
        "fit_rows": fit_rows, "early_stop_rows": stop_rows, "score_rows": len(predictions),
        "fit_ids_sha256": fit_ids, "early_stop_ids_sha256": stop_ids,
        "early_stop_rmse_seconds": stop_rmse, "max_model_reload_difference_seconds": max_reload_difference,
        "target_filter": "none", "prediction_postprocess": "none", "audit_used": False,
        "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run),
    }
    write_json(run / "run.json", report)
    emit("baseline_complete", model=kind, fold=fold, run=str(run), best_iterations=best,
         metrics=metrics["overall"], resources=monitor.stats)
    return run


def refit_run(root: Path, base: Path, config_path: Path, selection_path: Path,
              output: Path, label: str) -> Path:
    config, selection = read_json(config_path), read_json(selection_path)
    source_receipt = model_code_receipt(root)
    if selection["config_sha256"] != object_hash(config):
        raise ValueError("Selected configuration changed before refitting.")
    if selection["model_code_sha256"] != source_receipt["sha256"]:
        raise ValueError("Selected training code changed before refitting.")
    kind, iterations = selection["selected_model"], int(selection["refit_iterations"])
    if kind not in {"lightgbm", "catboost"} or iterations < 1:
        raise ValueError("Invalid selected refit recipe.")
    run = new_run(output, f"refit_{kind}")
    with ResourceMonitor() as monitor:
        manifest, catalog = load_bundle(root, base, config)
        if selection["bundle_id"] != manifest["bundle_id"]:
            raise ValueError("Selected feature bundle changed before refitting.")
        months = catalog[TIME].dt.strftime("%Y-%m")
        assignment = catalog[[ID]].copy()
        assignment["partition"] = np.where(months.isin(config["refit_months"]), "fit", "reserved")
        reserved = set(manifest["descriptor"]["protocol"]["reserved_audit_months"])
        if months.loc[assignment["partition"].eq("fit")].isin(reserved).any():
            raise ValueError("Refit includes reserved audit rows.")
        # Protect audit flight groups even if future data introduces boundary overlaps.
        protected_groups = set(catalog.loc[months.isin(reserved), "FLIGHT_ID_mvt"].dropna())
        assignment.loc[catalog["FLIGHT_ID_mvt"].isin(protected_groups), "partition"] = "reserved"
        features, truth = load_partition(base, catalog, assignment, "fit")
        encoder = fit_encoder(features, kind)
        x, y = transform(features, encoder), aligned_target(features, truth)
        fit_rows, fit_ids = len(x), identity_hash(features[ID])
        emit("refit_training", model=kind, fit_rows=fit_rows, iterations=iterations,
             fitting_months=config["refit_months"], audit_used=False, run=str(run))
        del features, truth
        gc.collect()
        model, actual, history, params = fit_model(kind, config, x, y, fixed_iterations=iterations)
        if actual != iterations:
            raise ValueError("Refit did not produce the selected number of trees.")
        model_path = save_model(model, kind, run, actual)
        write_json(run / "encoder.json", encoder)
        write_json(run / "config.json", config)
        write_json(run / "selection.json", selection)
        write_json(run / "training_history.json", history)
        del x, y
        gc.collect()
        ranking = pd.read_parquet(base / "features/ranking.parquet")
        encoded = transform(ranking, encoder)
        values = predict(model, encoded, kind, config["threads"])
        reloaded = load_model(model_path, kind)
        restored = predict(reloaded, encoded.iloc[:1000], kind, config["threads"])
        if not np.allclose(values[:len(restored)], restored, rtol=0, atol=1e-9):
            raise ValueError("Refit model persistence check failed.")
        predictions = pd.DataFrame({ID: ranking[ID], "prediction": values})
        if identity_hash(predictions[ID]) != manifest["ranking_ids_sha256"]:
            raise ValueError("Ranking prediction IDs changed.")
        predictions.to_parquet(run / "ranking_predictions.parquet", index=False, compression="zstd")
    report = {
        "run_id": run.name, "created_at_utc": utc_now(), "stage": "refit", "label": label,
        "model_kind": kind, "bundle_id": manifest["bundle_id"], "feature_code": manifest["code"],
        "model_code": source_receipt, "config_sha256": object_hash(config),
        "parameters": params, "iterations": iterations, "fit_rows": fit_rows,
        "fit_ids_sha256": fit_ids, "refit_months": config["refit_months"], "audit_used": False,
        "ranking_rows": len(predictions), "ranking_ids_sha256": identity_hash(predictions[ID]),
        "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run),
    }
    write_json(run / "run.json", report)
    emit("refit_complete", run=str(run), model=kind, ranking_rows=len(predictions),
         resources=monitor.stats)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--bundle", type=Path, default=Path("data/phase1/58d621b245de3f5e12dd"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase2"))
    parser.add_argument("--label", default="azure-arcw0")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--model", choices=["lightgbm", "catboost"], required=True)
    training.add_argument("--fold", choices=["seasonal", "forward", "final_audit"], required=True)
    refit = sub.add_parser("refit")
    refit.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "train":
        training_run(root, root / args.bundle, root / args.config, args.model, args.fold,
                     root / args.output, args.label)
    else:
        refit_run(root, root / args.bundle, root / args.config, root / args.selection,
                  root / args.output, args.label)
