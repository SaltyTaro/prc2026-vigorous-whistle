"""Complete-population residual models and self-contained specialist pipelines."""

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
from prc2026_phase3.training import fit_model, model_code_receipt as previous_model_code
from .bundle import load_bundle
from .encoding import aligned_target, contract, fit_encoder, transform
from .features import FEATURE_NAMES, POLICY, prediction_offset, route_mask


def model_code_receipt(root: Path) -> dict:
    paths = [*sorted((root / "src/prc2026_phase4").glob("*.py")),
             *sorted((root / "src/prc2026_models").glob("*.py")),
             root / "src/prc2026_phase3/training.py", root / "src/prc2026_phase3/bundle.py",
             root / "tools/run_phase4.py", root / "requirements-phase2.txt",
             root / "requirements-phase3.txt", root / "requirements-phase4.txt"]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def validate_config(config: dict, protocol: dict) -> None:
    if config["feature_policy"] != POLICY or config["validation_protocol"] != protocol["protocol_id"]:
        raise ValueError("Phase-4 feature/validation policy mismatch.")
    if config["target_filter"] != "none" or config["prediction_postprocess"] != "none":
        raise ValueError("No target filtering or prediction clipping is implemented.")
    if not 1 <= config["threads"] <= 16:
        raise ValueError("Use only the authorized 16-vCPU worker.")
    if config["development_folds"] != ["seasonal", "forward"]:
        raise ValueError("Both original development folds are required.")
    for fold in config["development_folds"]:
        require_development_fold(protocol, fold)
    if config["refit_months"] != [f"2025-{m:02}" for m in range(1, 12)]:
        raise ValueError("Refit must preserve January-November and exclude the reserved audit.")
    seen = set()
    for variant, spec in config["variants"].items():
        if spec["kind"] not in {"reference_gate", "global", "specialist", "blend"}:
            raise ValueError("Unknown model recipe.")
        if any(spec[k] not in seen for k in ["base", "source"] if k in spec):
            raise ValueError("Model dependencies must precede their consumers.")
        if "weight" in spec and not 0 <= spec["weight"] <= 1:
            raise ValueError("Specialist blend weights must be convex.")
        if spec.get("offset", "none") not in {"none", "lirf_missing_schedule"}:
            raise ValueError("Unknown offset policy.")
        seen.add(variant)


def model_settings(config: dict, spec: dict) -> dict:
    settings = {k: deepcopy(config[k]) for k in ["threads", "seed", "lightgbm"]}
    settings["lightgbm"]["parameters"].update(spec.get("parameters", {}))
    for key in ["rounds", "early_stopping_rounds"]:
        if key in spec:
            settings["lightgbm"][key] = spec[key]
    return settings


def node_prediction(run: Path, node: dict, features: pd.DataFrame, threads: int) -> np.ndarray:
    names, _ = contract(node["schema"])
    encoder = read_json(run / node["encoder"])
    encoded = transform(features[[ID, *names]], encoder)
    model = load_model(run / node["model"], "lightgbm")
    return predict(model, encoded, "lightgbm", threads) + prediction_offset(features, node["offset"])


def pipeline_prediction(run: Path, pipeline: dict, features: pd.DataFrame, threads: int) -> np.ndarray:
    nodes = pipeline["nodes"]
    values = node_prediction(run, nodes[pipeline["default"]], features, threads)
    for patch in pipeline["patches"]:
        mask = route_mask(features, patch["route"])
        if not mask.any():
            continue
        if patch["kind"] == "schedule":
            # Missing schedules fall back to the default learned model.
            mask = mask & np.isfinite(features["schedule_to_movement_s"].to_numpy(dtype=float))
            replacement = features.loc[mask, "schedule_to_movement_s"].to_numpy(dtype=float)
        elif patch["kind"] == "model":
            replacement = node_prediction(run, nodes[patch["node"]], features.loc[mask], threads)
        else:
            raise ValueError("Unknown prediction patch.")
        weight = patch["weight"]
        if not 0 <= weight <= 1:
            raise ValueError("Nonconvex specialist weight.")
        values[mask] = (1-weight)*values[mask] + weight*replacement
    if not np.isfinite(values).all() or values.shape != (len(features),):
        raise ValueError("Pipeline must predict every row exactly once.")
    return values


def fit_node(run: Path, name: str, schema: str, offset: str, route: str, settings: dict,
             features: pd.DataFrame, truth: pd.DataFrame,
             stop: pd.DataFrame | None = None, stop_truth: pd.DataFrame | None = None,
             iterations: int | None = None) -> dict:
    if (stop is None) != (stop_truth is None) or (iterations is not None and stop is not None):
        raise ValueError("A fixed refit cannot inspect early-stopping data.")
    names, _ = contract(schema)
    mask = route_mask(features, route)
    selected = features.loc[mask]
    if selected.empty:
        raise ValueError("Empty specialist fitting population.")
    selected_truth = truth.loc[truth[ID].isin(selected[ID])]
    projected = selected[[ID, *names]]
    encoder = fit_encoder(projected, schema)
    x = transform(projected, encoder)
    y = aligned_target(selected, selected_truth) - prediction_offset(selected, offset)
    x_stop = y_stop = selected_stop = None
    if stop is not None:
        selected_stop = stop.loc[route_mask(stop, route)]
        if selected_stop.empty:
            raise ValueError("Empty specialist early-stopping population.")
        x_stop = transform(selected_stop[[ID, *names]], encoder)
        y_stop = (aligned_target(selected_stop, stop_truth.loc[stop_truth[ID].isin(selected_stop[ID])])
                  - prediction_offset(selected_stop, offset))
    emit("phase4_component_training", component=name, route=route, rows=len(x), offset=offset,
         iterations=iterations, predictors=len(names))
    with ResourceMonitor() as monitor:
        model, best, history, parameters = fit_model(settings, encoder, x, y, x_stop, y_stop, iterations)
        if iterations is not None and best != iterations:
            raise ValueError("Refit did not produce the frozen number of trees.")
        paths = {"model": name + "_model.txt", "encoder": name + "_encoder.json",
                 "history": name + "_history.json", "importance": name + "_importance.json"}
        model.save_model(str(run / paths["model"]), num_iteration=best)
        write_json(run / paths["encoder"], encoder)
        write_json(run / paths["history"], history)
        write_json(run / paths["importance"], dict(zip(names, map(float, model.feature_importance(importance_type="gain")))))
        sample = (x if x_stop is None else x_stop).iloc[:1000]
        original_values = predict(model, sample, "lightgbm", settings["threads"])
        saved_values = predict(load_model(run / paths["model"], "lightgbm"), sample, "lightgbm", settings["threads"])
        difference = float(np.max(np.abs(original_values-saved_values)))
        if difference > 1e-9:
            raise ValueError("Persisted component prediction mismatch.")
        stop_rmse = None if x_stop is None else float(np.sqrt(np.mean((predict(model, x_stop, "lightgbm", settings["threads"])-y_stop)**2)))
    return {"schema": schema, "route": route, "offset": offset, "settings": settings,
            "iterations": best, "parameters": parameters, **paths,
            "fit_rows": len(selected), "fit_ids_sha256": identity_hash(selected[ID]),
            "early_stop_rows": 0 if selected_stop is None else len(selected_stop),
            "early_stop_ids_sha256": None if selected_stop is None else identity_hash(selected_stop[ID]),
            "early_stop_rmse_seconds": stop_rmse, "max_model_reload_difference_seconds": difference,
            "resources": monitor.stats}


def reference_runs(root: Path, path: Path, config: dict, original: dict) -> dict:
    found = {}
    expected_code = previous_model_code(root)["sha256"]
    for file in sorted(path.glob("*/run.json")):
        run = read_json(file)
        if run.get("stage") != "development" or run.get("variant") != config["reference_variant"]:
            continue
        fold = run["fold"]
        if fold not in config["development_folds"] or fold in found:
            raise ValueError("Missing/ambiguous reference fold.")
        if run["bundle_id"] != config["reference_bundle_id"] or run["model_code"]["sha256"] != expected_code or run["audit_used"]:
            raise ValueError("Reference model provenance mismatch.")
        for role in ["fit", "early_stop", "score"]:
            if run[role + "_ids_sha256"] != original["folds"][fold]["identity_sha256"][role]:
                raise ValueError("Reference split population changed.")
        verify_receipts(file.parent, run["artifacts"])
        found[fold] = (file.parent, run)
    if set(found) != set(config["development_folds"]):
        raise ValueError("Both Phase-3 reference folds are required.")
    return found


def import_reference(run: Path, reference: tuple[Path, dict]) -> dict:
    source, report = reference
    shutil.copyfile(source / "model.txt", run / "global_model.txt")
    encoder = read_json(source / "encoder.json")
    encoder["schema"] = "phase3"
    write_json(run / "global_encoder.json", encoder)
    settings = {k: read_json(source / "config.json")[k] for k in ["threads", "seed", "lightgbm"]}
    card = {"schema": "phase3", "route": "all", "offset": "none", "settings": settings,
            "iterations": report["best_iterations"], "model": "global_model.txt", "encoder": "global_encoder.json",
            "imported_from": {"run_id": report["run_id"], "run_sha256": sha256_file(source / "run.json")}}
    for key in ["fit_rows", "fit_ids_sha256", "early_stop_rows", "early_stop_ids_sha256", "early_stop_rmse_seconds",
                "max_model_reload_difference_seconds", "parameters"]:
        card[key] = report[key]
    return {"default": "global", "nodes": {"global": card}, "patches": []}


def import_pipeline(run: Path, source: Path) -> dict:
    report = read_json(source / "run.json")
    verify_receipts(source, report["artifacts"])
    pipeline = read_json(source / "pipeline.json")
    for node in pipeline["nodes"].values():
        for key in ["model", "encoder", "history", "importance"]:
            if key in node:
                shutil.copyfile(source / node[key], run / node[key])
    return pipeline


def development_run(root: Path, output: Path, config: dict, variant: str, fold: str,
                    manifest: dict, original: dict, frames: dict, truths: dict,
                    existing: dict, reference: tuple[Path, dict]) -> Path:
    spec, code = config["variants"][variant], model_code_receipt(root)
    run = new_run(output, f"{fold}_{variant}_lightgbm")
    with ResourceMonitor() as monitor:
        kind = spec["kind"]
        if kind == "reference_gate":
            pipeline = import_reference(run, reference)
            pipeline["patches"].append({"kind": "schedule", "route": "lirf_missing_record", "weight": 1.0})
        elif kind == "global":
            card = fit_node(run, "global", "phase4", spec["offset"], "all", model_settings(config, spec),
                            frames["fit"], truths["fit"], frames["early_stop"], truths["early_stop"])
            pipeline = {"default": "global", "nodes": {"global": card}, "patches": []}
        elif kind == "specialist":
            pipeline = import_pipeline(run, existing[spec["base"]])
            routes = ["airport:" + a for a in config["airports"]] if spec["routes"] == "all_airports" else spec["routes"]
            for route in routes:
                name = "specialist_" + route.replace(":", "_")
                card = fit_node(run, name, "phase4", "lirf_missing_schedule", route, model_settings(config, spec),
                                frames["fit"], truths["fit"], frames["early_stop"], truths["early_stop"])
                pipeline["nodes"][name] = card
                pipeline["patches"].append({"kind": "model", "route": route, "node": name, "weight": spec["weight"]})
        elif kind == "blend":
            pipeline = import_pipeline(run, existing[spec["source"]])
            for patch in pipeline["patches"]:
                patch["weight"] = spec["weight"]
        else:
            raise ValueError("Unknown recipe.")
        write_json(run / "pipeline.json", pipeline)
        write_json(run / "config.json", config)
        values = pipeline_prediction(run, pipeline, frames["score"], config["threads"])
        if kind == "reference_gate":
            saved = pd.read_parquet(reference[0] / "predictions.parquet")
            expected = frames["score"][[ID]].merge(saved, on=ID, validate="one_to_one")["prediction"].to_numpy()
            baseline = node_prediction(run, pipeline["nodes"]["global"], frames["score"], config["threads"])
            if not np.allclose(baseline, expected, rtol=0, atol=1e-9):
                raise ValueError("Imported reference model differs from its saved scoring predictions.")
        predictions = pd.DataFrame({ID: frames["score"][ID], "prediction": values})
        metrics = evaluate(truths["score"], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
        predictions.to_parquet(run / "predictions.parquet", compression="zstd", index=False)
        write_json(run / "metrics.json", metrics)
    report = {
        "created_at_utc": utc_now(), "run_id": run.name, "stage": "development", "model_kind": "lightgbm",
        "variant": variant, "fold": fold, "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
        "model_code": code, "feature_code": manifest["code"], "config_sha256": object_hash(config),
        "component_iterations": {n: node["iterations"] for n, node in pipeline["nodes"].items()},
        "max_model_reload_difference_seconds": max(n["max_model_reload_difference_seconds"] for n in pipeline["nodes"].values()),
        "audit_used": False, "target_filter": "none", "prediction_postprocess": "none",
        "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run),
    }
    for role in ["fit", "early_stop", "score"]:
        report[role + "_rows"], report[role + "_ids_sha256"] = len(frames[role]), identity_hash(frames[role][ID])
    write_json(run / "run.json", report)
    emit("phase4_development_complete", variant=variant, fold=fold, metrics=metrics["overall"],
         component_iterations=report["component_iterations"], resources=monitor.stats, run=str(run))
    return run


def suite(root: Path, base: Path, config_path: Path, output: Path, reference_path: Path, fold: str) -> None:
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    require_development_fold(original["descriptor"]["protocol"], fold)
    references = reference_runs(root, reference_path, config, original)
    code = model_code_receipt(root)
    existing = {}
    for file in output.glob("*/run.json"):
        report = read_json(file)
        if report.get("stage") != "development" or report.get("fold") != fold:
            continue
        if (report["config_sha256"] != object_hash(config) or report["model_code"]["sha256"] != code["sha256"]
                or report["bundle_id"] != manifest["bundle_id"] or report["audit_used"]):
            raise ValueError("Use a fresh output directory after model/configuration changes.")
        if report["variant"] in existing or report["variant"] not in config["variants"]:
            raise ValueError("Ambiguous saved variant.")
        verify_receipts(file.parent, report["artifacts"])
        existing[report["variant"]] = file.parent
    assignment = pd.read_parquet(parent / "splits" / f"{fold}.parquet")
    frames, truths = {}, {}
    for role in ["fit", "early_stop", "score"]:
        frames[role], truths[role] = load_partition(base, parent, catalog, assignment, role, FEATURE_NAMES)
        if identity_hash(frames[role][ID]) != original["folds"][fold]["identity_sha256"][role]:
            raise ValueError("Frozen fit/stop/score IDs changed.")
    for variant in config["variants"]:
        if variant in existing:
            emit("phase4_verified_completed_run", variant=variant, fold=fold, run=str(existing[variant]))
            continue
        existing[variant] = development_run(root, output, config, variant, fold, manifest, original,
                                             frames, truths, existing, references[fold])
        gc.collect()


def refit(root: Path, base: Path, config_path: Path, selection_path: Path, output: Path) -> Path:
    config, selection = read_json(config_path), read_json(selection_path)
    code = model_code_receipt(root)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    if (selection["config_sha256"] != object_hash(config) or selection["model_code_sha256"] != code["sha256"]
            or selection["bundle_id"] != manifest["bundle_id"] or selection["audit_used"]):
        raise ValueError("Selected model provenance changed.")
    variant = selection["selected_variant"]
    if not variant or variant not in config["variants"]:
        raise ValueError("No eligible development recipe for refit.")
    pipeline = deepcopy(selection["refit_pipeline"])
    run = new_run(output, f"refit_{variant}_lightgbm")
    with ResourceMonitor() as monitor:
        months = catalog[TIME].dt.strftime("%Y-%m")
        assignment = catalog[[ID]].copy()
        assignment["partition"] = np.where(months.isin(config["refit_months"]), "fit", "reserved")
        reserved = original["descriptor"]["protocol"]["reserved_audit_months"]
        protected = set(catalog.loc[months.isin(reserved), "FLIGHT_ID_mvt"].dropna())
        assignment.loc[catalog["FLIGHT_ID_mvt"].isin(protected), "partition"] = "reserved"
        features, truth = load_partition(base, parent, catalog, assignment, "fit", FEATURE_NAMES)
        fit_rows, fit_ids = len(features), identity_hash(features[ID])
        for name, node in list(pipeline["nodes"].items()):
            pipeline["nodes"][name] = fit_node(run, name, node["schema"], node["offset"], node["route"], node["settings"],
                                                features, truth, iterations=selection["refit_component_iterations"][name])
        write_json(run / "pipeline.json", pipeline)
        write_json(run / "config.json", config)
        write_json(run / "selection.json", selection)
        del features, truth
        gc.collect()
        ranking = pd.read_parquet(base / "features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
        values = pipeline_prediction(run, pipeline, ranking, config["threads"])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Refit ranking population changed.")
        pd.DataFrame({ID: ranking[ID], "prediction": values}).to_parquet(run / "ranking_predictions.parquet", index=False, compression="zstd")
    write_json(run / "run.json", {
        "created_at_utc": utc_now(), "run_id": run.name, "stage": "refit", "model_kind": "lightgbm",
        "variant": variant, "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
        "feature_code": manifest["code"], "model_code": code, "config_sha256": object_hash(config),
        "iterations": sum(selection["refit_component_iterations"].values()),
        "component_iterations": selection["refit_component_iterations"], "fit_rows": fit_rows, "fit_ids_sha256": fit_ids,
        "refit_months": config["refit_months"], "audit_used": False,
        "max_model_reload_difference_seconds": max(n["max_model_reload_difference_seconds"] for n in pipeline["nodes"].values()),
        "ranking_rows": len(ranking), "ranking_ids_sha256": identity_hash(ranking[ID]),
        "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run),
    })
    emit("phase4_refit_complete", variant=variant, run=str(run), ranking_rows=len(ranking), resources=monitor.stats)
    return run


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4"))
    parser.add_argument("--reference-runs", type=Path, default=Path("runs/phase3"))
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("suite")
    training.add_argument("--fold", choices=["seasonal", "forward", "final_audit"], required=True)
    fitting = sub.add_parser("refit")
    fitting.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    base = root / (args.bundle or read_json(root / "data/phase4/latest.json")["bundle"])
    if args.command == "suite":
        suite(root, base, root / args.config, root / args.output, root / args.reference_runs, args.fold)
    else:
        refit(root, base, root / args.config, root / args.selection, root / args.output)
