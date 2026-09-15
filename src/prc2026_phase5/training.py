"""Bounded component replacements, seed replicas, and frozen ensemble refits."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from prc2026.artifacts import (ResourceMonitor, object_hash, read_json, sha256_file,
                               utc_now, verify_receipts, write_json)
from prc2026.features import ID, TIME
from prc2026.phase1 import emit
from prc2026.validation import evaluate, identity_hash, require_development_fold
from prc2026_models.experiments import load_model, model_environment, new_run, predict, receipts
from prc2026_phase3.bundle import load_partition
from prc2026_phase4.bundle import load_bundle
from prc2026_phase4.encoding import aligned_target
from prc2026_phase4.features import FEATURE_NAMES, POLICY, prediction_offset, route_mask
from prc2026_phase4.training import fit_node as fit_lgb_node, model_code_receipt as parent_model_code
from .inference import cat_transform, fit_cat_encoder, iter_nodes, pipeline_prediction, validate_weights


def model_code_receipt(root: Path) -> dict:
    files = dict(parent_model_code(root)["files"])
    paths = [*sorted((root / "src/prc2026_phase5").glob("*.py")),
             root / "tools/run_phase5.py", root / "requirements-phase5.txt"]
    files.update({p.relative_to(root).as_posix(): sha256_file(p) for p in paths})
    return {"files": files, "sha256": object_hash(files)}


def validate_config(config: dict, protocol: dict) -> None:
    if config["feature_policy"] != POLICY or config["validation_protocol"] != protocol["protocol_id"] or config["feature_changes"]:
        raise ValueError("Phase 5 must use the frozen feature/validation policy.")
    if config["target_filter"] != "none" or config["prediction_postprocess"] != "none":
        raise ValueError("Target filtering and clipping are not implemented.")
    if not 1 <= config["threads"] <= 16:
        raise ValueError("Only one authorized 16-vCPU worker may be used.")
    if config["development_folds"] != ["seasonal", "forward"]:
        raise ValueError("Both protected development folds are required.")
    for fold in config["development_folds"]:
        require_development_fold(protocol, fold)
    if config["refit_months"] != [f"2025-{m:02}" for m in range(1, 12)]:
        raise ValueError("Refit must exclude the reserved audit and retain January-November.")
    order = config["variant_order"]
    if len(order) != len(set(order)) or set(order) != set(config["variants"]) or order[0] != "reference":
        raise ValueError("Explicit variant order must enumerate the batch once.")
    for spec in config["variants"].values():
        if spec["replace"] not in {"none", "global", "airports"}:
            raise ValueError("Unknown replacement scope.")
        if spec.get("kind", "lightgbm") not in {"lightgbm", "catboost"}:
            raise ValueError("Unknown native model family.")
        if spec.get("kind") == "catboost" and spec["parameters"].get("task_type") != "CPU":
            raise ValueError("This batch is CPU-only.")
    if len(config["seed_replicas"]) != 2 or len(set([config["seed"], *config["seed_replicas"]])) != 3:
        raise ValueError("Exactly two distinct additional seeds are declared.")
    if not 0 < config["minimum_improvement_fraction_each_fold"] < 1:
        raise ValueError("Promotion needs a positive improvement margin.")
    for weight in config["blend_policy"]["weights"]:
        validate_weights([weight, 1-weight])


def reference_runs(root: Path, runs: Path, config: dict, original: dict) -> dict:
    expected = parent_model_code(root)["sha256"]
    if expected != config["reference_model_code_sha256"]:
        raise ValueError("Frozen Phase-4 source changed.")
    found = {}
    for file in sorted(runs.glob("*/run.json")):
        run = read_json(file)
        if run["stage"] != "development" or run["variant"] != config["reference_variant"]:
            continue
        fold = run["fold"]
        if fold in found or fold not in config["development_folds"]:
            raise ValueError("Ambiguous Phase-4 reference.")
        if run["audit_used"] or run["bundle_id"] != config["feature_bundle_id"] or run["model_code"]["sha256"] != expected:
            raise ValueError("Reference provenance mismatch.")
        for role in ["fit", "early_stop", "score"]:
            if run[role + "_ids_sha256"] != original["folds"][fold]["identity_sha256"][role]:
                raise ValueError("Reference protected IDs changed.")
        verify_receipts(file.parent, run["artifacts"])
        found[fold] = (file.parent, run)
    if set(found) != set(config["development_folds"]):
        raise ValueError("Both Phase-4 reference folds are needed.")
    return found


def fit_cat_node(run: Path, name: str, blueprint: dict, settings: dict, features: pd.DataFrame,
                 truth: pd.DataFrame, stop: pd.DataFrame | None = None,
                 stop_truth: pd.DataFrame | None = None, iterations: int | None = None) -> dict:
    from catboost import CatBoostRegressor, Pool

    if (stop is None) != (stop_truth is None) or (iterations is not None and stop is not None):
        raise ValueError("A fixed refit cannot inspect stopping data.")
    if blueprint["schema"] != "phase4":
        raise ValueError("CatBoost expects the frozen Phase-4 feature contract.")
    route, offset = blueprint["route"], blueprint["offset"]
    selected = features.loc[route_mask(features, route)]
    labels = truth.loc[truth[ID].isin(selected[ID])]
    encoder = fit_cat_encoder(selected[[ID, *FEATURE_NAMES]])
    x = cat_transform(selected[[ID, *FEATURE_NAMES]], encoder)
    y = aligned_target(selected, labels)-prediction_offset(selected, offset)
    spec = settings["catboost"]
    params = dict(spec["parameters"])
    params.update(iterations=spec["rounds"] if iterations is None else iterations,
                  random_seed=settings["seed"], thread_count=settings["threads"])
    if params["iterations"] < 1 or params.get("task_type") != "CPU":
        raise ValueError("Invalid CatBoost iteration count or compute type.")
    selected_stop = x_stop = y_stop = None
    if stop is not None:
        selected_stop = stop.loc[route_mask(stop, route)]
        x_stop = cat_transform(selected_stop[[ID, *FEATURE_NAMES]], encoder)
        y_stop = (aligned_target(selected_stop, stop_truth.loc[stop_truth[ID].isin(selected_stop[ID])])
                  - prediction_offset(selected_stop, offset))
    emit("phase5_catboost_training", component=name, rows=len(x), rounds=params["iterations"], route=route)
    with ResourceMonitor() as monitor:
        model = CatBoostRegressor(**params)
        kwargs = {"verbose": 100}
        if x_stop is not None:
            kwargs.update(eval_set=Pool(x_stop, y_stop, cat_features=encoder["categorical_columns"]),
                          early_stopping_rounds=spec["early_stopping_rounds"], use_best_model=True)
        model.fit(Pool(x, y, cat_features=encoder["categorical_columns"]), **kwargs)
        best = int(model.tree_count_)
        if iterations is not None and best != iterations:
            raise ValueError("CatBoost refit tree count differs from the frozen recipe.")
        paths = {"model": name + "_model.cbm", "encoder": name + "_encoder.json",
                 "history": name + "_history.json", "importance": name + "_importance.json"}
        model.save_model(str(run / paths["model"]), format="cbm")
        write_json(run / paths["encoder"], encoder)
        write_json(run / paths["history"], model.get_evals_result())
        write_json(run / paths["importance"], dict(zip(FEATURE_NAMES, map(float, model.get_feature_importance()))))
        sample = (x if x_stop is None else x_stop).iloc[:1000]
        before = predict(model, sample, "catboost", settings["threads"])
        after = predict(load_model(run / paths["model"], "catboost"), sample, "catboost", settings["threads"])
        difference = float(np.max(np.abs(before-after)))
        if difference > 1e-9:
            raise ValueError("CatBoost persistence mismatch.")
        stopping_rmse = None if x_stop is None else float(np.sqrt(np.mean((predict(model, x_stop, "catboost", settings["threads"])-y_stop)**2)))
    return {"model_kind": "catboost", "schema": "phase4", "route": route, "offset": offset,
            "settings": settings, "parameters": params, "iterations": best, **paths,
            "fit_rows": len(selected), "fit_ids_sha256": identity_hash(selected[ID]),
            "early_stop_rows": 0 if selected_stop is None else len(selected_stop),
            "early_stop_ids_sha256": None if selected_stop is None else identity_hash(selected_stop[ID]),
            "early_stop_rmse_seconds": stopping_rmse, "max_model_reload_difference_seconds": difference,
            "resources": monitor.stats}


def fit_component(run, name, blueprint, settings, features, truth, stop=None, stop_truth=None, iterations=None):
    if blueprint.get("model_kind", "lightgbm") == "catboost":
        return fit_cat_node(run, name, blueprint, settings, features, truth, stop, stop_truth, iterations)
    node = fit_lgb_node(run, name, blueprint["schema"], blueprint["offset"], blueprint["route"], settings,
                        features, truth, stop, stop_truth, iterations)
    node["model_kind"] = "lightgbm"
    return node


def derive_settings(config: dict, original: dict, spec: dict) -> dict:
    kind = spec.get("kind", original.get("model_kind", "lightgbm"))
    if kind == "catboost":
        return {"threads": config["threads"], "seed": config["seed"], "catboost": {
            "rounds": spec["rounds"], "early_stopping_rounds": spec["early_stopping_rounds"],
            "parameters": deepcopy(spec["parameters"])}}
    settings = deepcopy(original["settings"])
    settings.update(threads=config["threads"], seed=config["seed"])
    settings["lightgbm"]["parameters"].update(spec.get("parameters", {}))
    for key in ["rounds", "early_stopping_rounds"]:
        if key in spec:
            settings["lightgbm"][key] = spec[key]
    return settings


def checked_runs(root: Path, output: Path, config: dict, manifest: dict) -> dict:
    found = {}
    code = model_code_receipt(root)["sha256"]
    for file in sorted(output.glob("*/run.json")):
        run = read_json(file)
        if run["stage"] != "development":
            continue
        key = (run["variant"], run["fold"])
        if key in found or run["audit_used"] or run["bundle_id"] != manifest["bundle_id"]:
            raise ValueError("Ambiguous development provenance.")
        if run["model_code"]["sha256"] != code or run["config_sha256"] != object_hash(config):
            raise ValueError("Use a fresh output directory after model/configuration changes.")
        verify_receipts(file.parent, run["artifacts"])
        found[key] = (file.parent, run)
    return found


def development_run(root: Path, output: Path, config: dict, variant: str, fold: str,
                    manifest: dict, original: dict, frames: dict, truths: dict,
                    source: Path, spec: dict | None = None, seed: int | None = None) -> Path:
    source_report = read_json(source / "run.json")
    verify_receipts(source, source_report["artifacts"])
    pipeline = deepcopy(read_json(source / "pipeline.json"))
    run = new_run(output, f"{fold}_{variant}")
    lineage = {"run_id": source_report["run_id"], "run_sha256": sha256_file(source / "run.json")}
    with ResourceMonitor() as monitor:
        for name, node in list(pipeline["nodes"].items()):
            replace = seed is not None or (spec["replace"] == "global" and name == pipeline["default"]) or (
                spec["replace"] == "airports" and name != pipeline["default"])
            if replace:
                blueprint = deepcopy(node)
                if seed is not None:
                    settings = deepcopy(node["settings"])
                    settings.update(seed=seed, threads=config["threads"])
                else:
                    blueprint["model_kind"] = spec["kind"]
                    settings = derive_settings(config, node, spec)
                pipeline["nodes"][name] = fit_component(run, name, blueprint, settings, frames["fit"], truths["fit"],
                                                        frames["early_stop"], truths["early_stop"])
            else:
                for key in ["model", "encoder", "history", "importance"]:
                    if key in node:
                        shutil.copyfile(source / node[key], run / node[key])
                node["model_kind"] = node.get("model_kind", "lightgbm")
                node["copied_from"] = lineage
        write_json(run / "pipeline.json", pipeline)
        write_json(run / "config.json", config)
        values = pipeline_prediction(run, pipeline, frames["score"], config["threads"])
        predictions = pd.DataFrame({ID: frames["score"][ID], "prediction": values})
        metrics = evaluate(truths["score"], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
        if variant == "reference":
            old = pd.read_parquet(source / "predictions.parquet")
            expected = frames["score"][[ID]].merge(old, on=ID, validate="one_to_one")["prediction"].to_numpy()
            if not np.allclose(values, expected, rtol=0, atol=1e-9):
                raise ValueError("The imported v3 pipeline must reproduce its development predictions.")
        predictions.to_parquet(run / "predictions.parquet", index=False, compression="zstd")
        write_json(run / "metrics.json", metrics)
    report = {"created_at_utc": utc_now(), "run_id": run.name, "stage": "development", "model_kind": "ensemble",
              "variant": variant, "fold": fold, "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
              "model_code": model_code_receipt(root), "config_sha256": object_hash(config), "source_run": lineage,
              "recipe": spec, "seed_override": seed,
              "component_iterations": {n: d["iterations"] for n, d in iter_nodes(pipeline)},
              "audit_used": False, "target_filter": "none", "prediction_postprocess": "none",
              "max_model_reload_difference_seconds": max(n["max_model_reload_difference_seconds"] for _, n in iter_nodes(pipeline)),
              "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run)}
    for role in ["fit", "early_stop", "score"]:
        report[role + "_rows"], report[role + "_ids_sha256"] = len(frames[role]), identity_hash(frames[role][ID])
    write_json(run / "run.json", report)
    emit("phase5_development_complete", variant=variant, fold=fold, metrics=metrics["overall"], resources=monitor.stats, run=str(run))
    return run


def suite(root: Path, base: Path, config_path: Path, output: Path, reference_path: Path,
          fold: str, seed_plan_path: Path | None = None):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    require_development_fold(original["descriptor"]["protocol"], fold)
    if manifest["bundle_id"] != config["feature_bundle_id"]:
        raise ValueError("The Phase-4 feature bundle changed.")
    references = reference_runs(root, reference_path, config, original)
    existing = checked_runs(root, output, config, manifest)
    seed_plan = None
    if seed_plan_path:
        seed_plan = read_json(seed_plan_path)
        if (seed_plan["config_sha256"] != object_hash(config) or seed_plan["model_code_sha256"] != model_code_receipt(root)["sha256"]
                or seed_plan["bundle_id"] != manifest["bundle_id"] or seed_plan["audit_used"]):
            raise ValueError("Seed plan provenance changed.")
        anchor = seed_plan["anchor_variant"]
        source, source_report = existing[(anchor, fold)]
        if sha256_file(source / "run.json") != seed_plan["anchor_runs"][fold]["run_sha256"]:
            raise ValueError("Seed anchor run changed.")
    assignment = pd.read_parquet(parent / "splits" / f"{fold}.parquet")
    frames, truths = {}, {}
    for role in ["fit", "early_stop", "score"]:
        frames[role], truths[role] = load_partition(base, parent, catalog, assignment, role, FEATURE_NAMES)
        if identity_hash(frames[role][ID]) != original["folds"][fold]["identity_sha256"][role]:
            raise ValueError("Protected fit/stop/score IDs changed.")
    jobs = [(f"seed_{seed}", None, seed) for seed in config["seed_replicas"]] if seed_plan else [
        (v, config["variants"][v], None) for v in config["variant_order"]]
    for variant, spec, seed in jobs:
        if (variant, fold) in existing:
            emit("phase5_verified_completed_run", variant=variant, fold=fold)
            continue
        development_run(root, output, config, variant, fold, manifest, original, frames, truths,
                        source if seed_plan else references[fold][0], spec, seed)
        gc.collect()


def refit(root: Path, base: Path, config_path: Path, selection_path: Path, output: Path) -> Path:
    config, selection = read_json(config_path), read_json(selection_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    if (selection["config_sha256"] != object_hash(config) or selection["model_code_sha256"] != model_code_receipt(root)["sha256"]
            or selection["bundle_id"] != manifest["bundle_id"] or selection["audit_used"] or not selection["selected_candidate"]):
        raise ValueError("No eligible, unchanged recipe for refit.")
    pipeline = deepcopy(selection["refit_pipeline"])
    run = new_run(output, "refit_" + selection["selected_candidate"])
    with ResourceMonitor() as monitor:
        months = catalog[TIME].dt.strftime("%Y-%m")
        assignment = catalog[[ID]].copy()
        assignment["partition"] = np.where(months.isin(config["refit_months"]), "fit", "reserved")
        protected = set(catalog.loc[months.isin(original["descriptor"]["protocol"]["reserved_audit_months"]), "FLIGHT_ID_mvt"].dropna())
        assignment.loc[catalog["FLIGHT_ID_mvt"].isin(protected), "partition"] = "reserved"
        features, truth = load_partition(base, parent, catalog, assignment, "fit", FEATURE_NAMES)
        fit_rows, fit_ids = len(features), identity_hash(features[ID])
        fitted = {}
        for qualified, node in iter_nodes(pipeline):
            iterations = selection["refit_component_iterations"][qualified]
            recipe = {k: node[k] for k in ["schema", "route", "offset", "settings"]}
            recipe.update(model_kind=node.get("model_kind", "lightgbm"), iterations=iterations)
            digest = object_hash(recipe)
            if digest not in fitted:
                fitted[digest] = fit_component(run, "component_" + digest[:16], node, node["settings"], features, truth, iterations=iterations)
            node.clear()
            node.update(deepcopy(fitted[digest]))
        write_json(run / "pipeline.json", pipeline)
        write_json(run / "selection.json", selection)
        write_json(run / "config.json", config)
        del features, truth
        gc.collect()
        ranking = pd.read_parquet(base / "features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
        values = pipeline_prediction(run, pipeline, ranking, config["threads"])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking population changed.")
        pd.DataFrame({ID: ranking[ID], "prediction": values}).to_parquet(run / "ranking_predictions.parquet", index=False, compression="zstd")
    write_json(run / "run.json", {
        "created_at_utc": utc_now(), "run_id": run.name, "stage": "refit", "model_kind": "ensemble",
        "variant": selection["selected_candidate"], "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
        "model_code": model_code_receipt(root), "config_sha256": object_hash(config), "audit_used": False,
        "iterations": sum(n["iterations"] for n in fitted.values()), "unique_components": len(fitted),
        "component_iterations": selection["refit_component_iterations"], "fit_rows": fit_rows, "fit_ids_sha256": fit_ids,
        "refit_months": config["refit_months"], "ranking_rows": len(ranking), "ranking_ids_sha256": identity_hash(ranking[ID]),
        "max_model_reload_difference_seconds": max(n["max_model_reload_difference_seconds"] for n in fitted.values()),
        "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run)})
    emit("phase5_refit_complete", run=str(run), ranking_rows=len(ranking), unique_components=len(fitted), resources=monitor.stats)
    return run


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/phase4/2ceb3449c26767a97311"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase5"))
    parser.add_argument("--reference-runs", type=Path, default=Path("runs/phase4"))
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("suite")
    training.add_argument("--fold", choices=["seasonal", "forward", "final_audit"], required=True)
    training.add_argument("--seed-plan", type=Path)
    fitting = sub.add_parser("refit")
    fitting.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "suite":
        suite(root, root / args.bundle, root / args.config, root / args.output, root / args.reference_runs,
              args.fold, None if args.seed_plan is None else root / args.seed_plan)
    else:
        refit(root, root / args.bundle, root / args.config, root / args.selection, root / args.output)
