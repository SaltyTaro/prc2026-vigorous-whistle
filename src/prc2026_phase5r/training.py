"""Bounded, receipt-backed model comparisons and an audit-preserving refit."""

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
from prc2026_phase5.inference import combine_predictions
from prc2026_phase5.training import model_code_receipt as previous_model_code
from .baseline import aligned_truth, apply_map, center_scale, cross_fit
from .bundle import load_bundle
from .encoding import assemble, contract, fit_encoder, transform
from .features import FEATURE_NAMES, POLICY
from .inference import airports_prediction, copy_legacy_pipeline, node_prediction, pipeline_prediction, route_mask


def model_code_receipt(root: Path) -> dict:
    hashes = dict(previous_model_code(root)["files"])
    paths = [root/"src/prc2026_phase5r"/name for name in ["__init__.py", "baseline.py", "encoding.py", "inference.py", "training.py"]]
    paths += [root/"tools/run_phase5r.py", root/"requirements-phase5r.txt"]
    hashes.update({p.relative_to(root).as_posix(): sha256_file(p) for p in paths})
    return {"files": hashes, "sha256": object_hash(hashes)}


def validate_config(config: dict, protocol: dict):
    if config["feature_policy"] != POLICY or config["validation_protocol"] != protocol["protocol_id"]:
        raise ValueError("New feature/validation policy mismatch.")
    if config["target_filter"] != "none" or config["prediction_postprocess"] != "none" or config["audit_used"]:
        raise ValueError("No clipping, target filtering or audit selection is permitted.")
    if config["refit_months"] != [f"2025-{m:02}" for m in range(1, 12)]:
        raise ValueError("Refit must exclude the reserved audit period.")
    if config["development_folds"] != ["seasonal", "forward"] or not 1 <= config["threads"] <= 16:
        raise ValueError("Use both original folds and only the authorized worker.")
    for fold in config["development_folds"]:
        require_development_fold(protocol, fold)
    if set(config["variant_order"]) != set(config["variants"]) or len(config["variant_order"]) != len(config["variants"]):
        raise ValueError("Declared variant order must be complete and unique.")
    for spec in config["variants"].values():
        contract(spec)
        if spec["route"] not in {"all", "nm_missing"}:
            raise ValueError("Unknown specialist route.")
    if any(not 0 < w <= 1 for w in config["blend_weights_new_candidate"]):
        raise ValueError("Invalid common candidate blend weights.")


def settings_for(config: dict, spec: dict) -> dict:
    settings = {k: deepcopy(config[k]) for k in ["threads", "seed", "lightgbm", "baseline"]}
    settings["lightgbm"]["parameters"].update(spec.get("parameters", {}))
    for name in ["rounds", "early_stopping_rounds"]:
        if name in spec:
            settings["lightgbm"][name] = spec[name]
    return settings


def fit_node(run: Path, name: str, spec: dict, settings: dict, features: pd.DataFrame, truth: pd.DataFrame,
             stop: pd.DataFrame | None = None, stop_truth: pd.DataFrame | None = None, iterations: int | None = None) -> dict:
    import lightgbm as lgb
    if (stop is None) != (stop_truth is None) or (iterations is not None and stop is not None):
        raise ValueError("A fixed refit cannot use stopping data; stopping inputs must be paired.")
    if iterations is not None and iterations < 1:
        raise ValueError("Positive fixed iterations required.")
    mask = route_mask(features, spec["route"])
    fit = features.loc[mask].reset_index(drop=True)
    labels = truth.loc[truth[ID].isin(fit[ID])]
    aligned = aligned_truth(fit, labels)
    fit_ids = identity_hash(fit[ID])
    paths = {"model": f"{name}_model.txt", "encoder": f"{name}_encoder.json", "history": f"{name}_history.json",
             "importance": f"{name}_importance.json", "baseline": f"{name}_baseline.json"}
    if any((run/p).exists() for p in paths.values()):
        raise ValueError("Refusing to overwrite a fitted component.")
    with ResourceMonitor() as monitor:
        baseline, full_map, cross_receipt = None, None, None
        if spec["target"] != "none":
            emit("phase5r_crossfit", component=name, rows=len(fit), inner_folds=settings["baseline"]["inner_folds"])
            baseline, full_map, cross_receipt, assignment = cross_fit(fit, labels, settings["baseline"])
            saved = pd.concat([assignment, baseline], axis=1)
            saved.to_parquet(run/f"{name}_fit_baseline.parquet", index=False, compression="zstd")
            write_json(run/paths["baseline"], full_map)
            write_json(run/f"{name}_crossfit.json", cross_receipt)
        assembled = assemble(fit, spec, baseline)
        encoder = fit_encoder(assembled, spec)
        x = transform(assembled, encoder)
        center, scale, weights = center_scale(baseline, spec["target"], len(fit))
        y_seconds = aligned[TARGET].to_numpy(dtype=float)
        y = (y_seconds-prediction_offset(fit, "lirf_missing_schedule")-center)/scale
        params = dict(settings["lightgbm"]["parameters"])
        params.update(num_threads=settings["threads"], seed=settings["seed"], feature_fraction_seed=settings["seed"],
                      bagging_seed=settings["seed"], data_random_seed=settings["seed"], metric="None")
        dataset = lgb.Dataset(x, label=y, weight=weights, categorical_feature=encoder["categorical_columns"])
        history, kwargs, stop_ids, stop_rows = {}, {}, None, 0
        callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history)]
        if stop is not None:
            stop_mask = route_mask(stop, spec["route"])
            stopped = stop.loc[stop_mask].reset_index(drop=True)
            stop_labels = stop_truth.loc[stop_truth[ID].isin(stopped[ID])]
            stop_aligned = aligned_truth(stopped, stop_labels)
            stop_baseline = apply_map(stopped, full_map) if full_map else None
            xs = transform(assemble(stopped, spec, stop_baseline), encoder)
            sc, ss, sw = center_scale(stop_baseline, spec["target"], len(stopped))
            sy = (stop_aligned[TARGET].to_numpy(dtype=float)-prediction_offset(stopped, "lirf_missing_schedule")-sc)/ss
            stop_dataset = lgb.Dataset(xs, label=sy, weight=sw, reference=dataset, categorical_feature=encoder["categorical_columns"])

            def seconds_rmse(values, data):
                if len(values) != len(ss):
                    raise ValueError("Unexpected early-stop population.")
                return "seconds_rmse", float(np.sqrt(np.mean(((values-sy)*ss)**2))), False

            kwargs = {"valid_sets": [stop_dataset], "valid_names": ["early_stop"], "feval": seconds_rmse}
            callbacks.append(lgb.early_stopping(settings["lightgbm"]["early_stopping_rounds"], first_metric_only=True))
            stop_ids, stop_rows = identity_hash(stopped[ID]), len(stopped)
        rounds = settings["lightgbm"]["rounds"] if iterations is None else iterations
        emit("phase5r_training", component=name, route=spec["route"], target=spec["target"], rows=len(fit), predictors=len(encoder["feature_names"]))
        model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=callbacks, **kwargs)
        best = int(model.best_iteration or model.current_iteration())
        if iterations is not None and best != iterations:
            raise ValueError("Refit differs from the frozen iteration recipe.")
        model.save_model(str(run/paths["model"]), num_iteration=best)
        write_json(run/paths["encoder"], encoder)
        write_json(run/paths["history"], history)
        write_json(run/paths["importance"], dict(zip(encoder["feature_names"], map(float, model.feature_importance(importance_type="gain")))))
        card = {"model_kind": "lightgbm", "schema": "phase5r", "spec": spec, "settings": settings,
                "parameters": params, "iterations": best, "model_path": paths["model"], "encoder_path": paths["encoder"],
                "baseline_path": paths["baseline"] if full_map else None,
                "fit_rows": len(fit), "fit_ids_sha256": fit_ids, "early_stop_rows": stop_rows, "early_stop_ids_sha256": stop_ids,
                "crossfit": cross_receipt, "target_filter": "none"}
        sample = (stopped if stop is not None else fit).iloc[:1000]
        sample_baseline = apply_map(sample, full_map) if full_map else None
        sample_x = transform(assemble(sample, spec, sample_baseline), encoder)
        c, s, _ = center_scale(sample_baseline, spec["target"], len(sample))
        expected = predict(model, sample_x, "lightgbm", settings["threads"])*s+c+prediction_offset(sample, "lirf_missing_schedule")
        restored = node_prediction(run, card, sample, settings["threads"])
        difference = float(np.max(np.abs(expected-restored)))
        if difference > 1e-9:
            raise ValueError("Persisted target transform/model does not reproduce predictions.")
        card["max_model_reload_difference_seconds"] = difference
        if stop is not None:
            card["early_stop_rmse_seconds"] = float(np.sqrt(np.mean(((predict(model, xs, "lightgbm", settings["threads"])-sy)*ss)**2)))
    card["resources"] = monitor.stats
    return card


def reference_selection(root: Path, config: dict) -> dict:
    path = root/config["reference_selection"]
    if sha256_file(path) != config["reference_selection_sha256"]:
        raise ValueError("The frozen v4 selection receipt changed.")
    selection = read_json(path)
    if selection["audit_used"] or selection["model_code_sha256"] != previous_model_code(root)["sha256"]:
        raise ValueError("Frozen reference code/audit policy changed.")
    return selection


def reference_predictions(root: Path, config: dict, fold: str, features: pd.DataFrame,
                          prior_runs: Path, followup_runs: Path):
    selection = reference_selection(root, config)
    weights = selection["candidates"][selection["selected_candidate"]]["weights"]
    parts, sources, paths = {}, {}, {}
    for name in weights:
        declared = selection["initial_and_seed_runs"][name]["folds"][fold]
        directory = (followup_runs if declared["run_group"] == "followup" else prior_runs)/declared["run_id"]
        report = read_json(directory/"run.json")
        if sha256_file(directory/"run.json") != declared["run_sha256"] or report["audit_used"]:
            raise ValueError("Reference development run changed.")
        verify_receipts(directory, report["artifacts"])
        parts[name] = pd.read_parquet(directory/"predictions.parquet")
        paths[name] = directory
        sources[name] = {"run_id": directory.name, "run_sha256": declared["run_sha256"],
                         "predictions_sha256": sha256_file(directory/"predictions.parquet"), "weight": weights[name]}
    mixed = combine_predictions(parts, weights)
    mixed = features[[ID]].merge(mixed, on=ID, how="left", validate="one_to_one")
    if mixed["prediction"].isna().any() or len(mixed) != len(parts[next(iter(parts))]):
        raise ValueError("Reference scoring identity changed.")
    old_path = paths["global_capacity"]
    airport = airports_prediction(old_path, read_json(old_path/"pipeline.json"), features, config["threads"])
    return mixed, pd.DataFrame({ID: features[ID], "prediction": airport}), sources


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
            emit("phase5r_run_reused", variant=variant, fold=fold)
            continue
        spec = config["variants"][variant]
        run = new_run(output, f"{fold}_{variant}")
        with ResourceMonitor() as monitor:
            node = fit_node(run, "new", spec, settings_for(config, spec), frames["fit"], truths["fit"],
                            frames["early_stop"], truths["early_stop"])
            mask = route_mask(frames["score"], spec["route"])
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
        emit("phase5r_run_complete", variant=variant, fold=fold, iterations=node["iterations"], metrics=metrics["overall"], resources=monitor.stats)
        gc.collect()


def refit(root: Path, base: Path, config_path: Path, output: Path, selection_path: Path, prior_runs: Path):
    config, selection = read_json(config_path), read_json(selection_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    if (selection["config_sha256"] != object_hash(config) or selection["model_code_sha256"] != model_code_receipt(root)["sha256"]
            or selection["bundle_id"] != manifest["bundle_id"] or selection["audit_used"] or selection["selected_candidate"] is None):
        raise ValueError("No unchanged, eligible recipe for refit.")
    candidate = selection["candidates"][selection["selected_candidate"]]
    if not candidate["eligible"]:
        raise ValueError("Cannot refit an ineligible development recipe.")
    variant, weight = candidate["variant"], candidate["new_weight"]
    spec = config["variants"][variant]
    iterations = selection["refit_iterations"]
    source = prior_runs/config["reference_refit_run"]
    source_report = read_json(source/"run.json")
    if sha256_file(source/"run.json") != config["reference_refit_sha256"] or source_report["audit_used"]:
        raise ValueError("Frozen v4 refit identity changed.")
    verify_receipts(source, source_report["artifacts"])
    run = new_run(output, f"refit_{selection['selected_candidate']}")
    with ResourceMonitor() as monitor:
        months = catalog[TIME].dt.strftime("%Y-%m")
        assignment = catalog[[ID]].copy()
        assignment["partition"] = np.where(months.isin(config["refit_months"]), "fit", "reserved")
        protected = set(catalog.loc[months.isin(original["descriptor"]["protocol"]["reserved_audit_months"]), "FLIGHT_ID_mvt"].dropna())
        assignment.loc[catalog["FLIGHT_ID_mvt"].isin(protected), "partition"] = "reserved"
        features, truth = load_partition(base, parent, catalog, assignment, "fit", FEATURE_NAMES)
        node = fit_node(run, "new", spec, settings_for(config, spec), features, truth, iterations=iterations)
        new = {"kind": "new_node", "node": node}
        # Imported native models remain byte-identical; only the new node is
        # trained. This final artifact is self-contained for reproduction.
        old = copy_legacy_pipeline(run, source, "v4")
        legacy = {"kind": "legacy", "pipeline": old}
        if spec["route"] == "nm_missing":
            pipeline = {"kind": "route_patch", "base": legacy, "replacement": new, "route": "nm_missing", "weight": weight}
        else:
            first = next(m["pipeline"] for m in old["members"] if m["name"] == "catboost_long")
            airport_only = {"kind": "legacy_airports", "pipeline": first}
            proposed = {"kind": "mixture", "members": [{"weight": .5, "pipeline": new}, {"weight": .5, "pipeline": airport_only}]}
            pipeline = proposed if weight == 1 else {"kind": "mixture", "members": [
                {"weight": weight, "pipeline": proposed}, {"weight": 1-weight, "pipeline": legacy}]}
        write_json(run/"pipeline.json", pipeline)
        write_json(run/"node.json", node)
        write_json(run/"selection.json", selection)
        write_json(run/"config.json", config)
        ranking = pd.read_parquet(base/"features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking identity changed.")
        values = pipeline_prediction(run, pipeline, ranking, config["threads"])
        predictions = pd.DataFrame({ID: ranking[ID], "prediction": values})
        predictions.to_parquet(run/"ranking_predictions.parquet", index=False, compression="zstd")
        restored = pipeline_prediction(run, read_json(run/"pipeline.json"), ranking.iloc[:1000], config["threads"])
        difference = float(np.max(np.abs(values[:1000]-restored)))
        if difference > 1e-8:
            raise ValueError("Persisted final pipeline differs from ranking predictions.")
    report = {"created_at_utc": utc_now(), "run_id": run.name, "stage": "refit", "model_kind": "ensemble",
              "variant": selection["selected_candidate"], "iterations": iterations, "bundle_id": manifest["bundle_id"],
              "parent_bundle_id": original["bundle_id"], "feature_code": manifest["code"], "model_code": model_code_receipt(root),
              "config_sha256": object_hash(config), "selection_sha256": sha256_file(selection_path),
              "fit_rows": len(features), "fit_ids_sha256": identity_hash(features[ID]), "refit_months": config["refit_months"],
              "audit_used": False, "target_filter": "none", "prediction_postprocess": "none", "ranking_rows": len(ranking),
              "ranking_ids_sha256": identity_hash(ranking[ID]), "max_model_reload_difference_seconds": difference,
              "source_refit_sha256": config["reference_refit_sha256"], "environment": model_environment(root),
              "resources": monitor.stats, "artifacts": receipts(run)}
    write_json(run/"run.json", report)
    emit("phase5r_refit_complete", run=str(run), rows=len(ranking), resources=monitor.stats)
    return run


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5r_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase5r"))
    parser.add_argument("--prior-runs", type=Path, default=Path("runs/phase5"))
    parser.add_argument("--followup-runs", type=Path, default=Path("runs/phase5_followup"))
    sub = parser.add_subparsers(dest="command", required=True)
    dev = sub.add_parser("suite")
    dev.add_argument("--fold", required=True, choices=["seasonal", "forward", "final_audit"])
    dev.add_argument("--variant", action="append")
    final = sub.add_parser("refit")
    final.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/"data/phase5r/latest.json")["bundle"])
    if args.command == "suite":
        suite(root, base, root/args.config, root/args.output, args.fold, root/args.prior_runs, root/args.followup_runs, args.variant)
    else:
        refit(root, base, root/args.config, root/args.output, root/args.selection, root/args.prior_runs)
