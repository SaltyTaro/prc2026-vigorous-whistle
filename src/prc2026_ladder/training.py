"""Eight declared native fits against the frozen v6 development predictions."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from prc2026.artifacts import (
    ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json,
)
from prc2026.features import ID
from prc2026.phase1 import emit
from prc2026.validation import evaluate, identity_hash, require_development_fold
from prc2026_models.experiments import model_environment, new_run, receipts
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_phase5.inference import validate_weights
from prc2026_nmclock.bundle import load_bundle
from prc2026_nmclock.features import FEATURE_NAMES, MODEL_NAMES
from prc2026_nmclock.encoding import contract as clock_contract
from prc2026_nmclock.inference import node_prediction as old_node_prediction, pipeline_prediction as old_pipeline_prediction
from prc2026_nmclock.training import fit_node as fit_clock, model_code_receipt as parent_code
from prc2026_weather.encoding import contract as weather_contract
from prc2026_weather.training import fit_node as fit_weather
from .inference import node_prediction, route_mask

REFERENCE_NAMES = {"v6", "v6_global", "v6_airports"}


def model_code_receipt(root):
    hashes = dict(parent_code(root)["files"])
    paths = [root/"src/prc2026_ladder"/name for name in ["__init__.py", "inference.py", "training.py"]]
    paths += [root/"tools/run_ladder.py", root/"requirements-ladder.txt"]
    hashes.update({path.relative_to(root).as_posix(): sha256_file(path) for path in paths})
    return {"files": hashes, "sha256": object_hash(hashes)}


def native_spec(variant):
    return {name: deepcopy(variant[name]) for name in ["kind", "features", "route", "target"] if name in variant}


def validate_config(config, protocol):
    if config["validation_protocol"] != protocol["protocol_id"] or config["development_folds"] != ["seasonal", "forward"]:
        raise ValueError("Both original development folds are required.")
    if config["audit_used"] or config["target_filter"] != "none" or config["prediction_postprocess"] != "none":
        raise ValueError("No audit, target filtering or clipping is allowed.")
    if config["refit_months"] != [f"2025-{month:02}" for month in range(1, 12)]:
        raise ValueError("December must remain excluded from refitting.")
    if config["new_shared_features"] or not 1 <= config["threads"] <= 16:
        raise ValueError("This campaign reuses frozen features on the authorized worker.")
    if len(config["variant_order"]) != 4 or set(config["variant_order"]) != set(config["variants"]):
        raise ValueError("The four declared variants must be complete and unique.")
    for fold in config["development_folds"]:
        require_development_fold(protocol, fold)
    for name, variant in config["variants"].items():
        spec = native_spec(variant)
        if variant["schema"] == "nmclock":
            names, _ = clock_contract(spec)
            if spec["route"] != "clock_available" or spec["target"] != "nm_residual":
                raise ValueError("Clock models retain the declared residual target and route.")
        elif variant["schema"] == "weather":
            names, _ = weather_contract(spec)
            if spec["route"] != "clock_missing":
                raise ValueError("Fallback experts fit only missing-clock rows.")
        else:
            raise ValueError("Unknown native schema.")
        if names != config["feature_contracts"][spec["features"]]:
            raise ValueError("The frozen feature contract changed.")
        settings = variant["settings"]
        if settings["threads"] != config["threads"] or settings["seed"] != config["seed"]:
            raise ValueError("Declared CPU settings and seeds must agree.")
        if spec["kind"] not in {"catboost", "lightgbm"} or settings[spec["kind"]]["rounds"] < 1:
            raise ValueError("Unsupported native model or iteration cap.")
        if spec["kind"] == "catboost" and settings["catboost"]["parameters"]["task_type"] != "CPU":
            raise ValueError("Only CPU CatBoost is authorized.")
    for family in config["candidate_families"].values():
        if set(family) != {"clock_available", "clock_missing"}:
            raise ValueError("Every family must cover both complementary routes.")
        for route, weights in family.items():
            validate_weights(list(weights.values()))
            if any(value <= 0 for value in weights.values()):
                raise ValueError("Declared family members must have positive weights.")
            for name in weights:
                if name in {"v6_global", "v6_airports"} and route != "clock_available":
                    raise ValueError("Imported clock components cannot predict missing clocks.")
                if name not in REFERENCE_NAMES:
                    if name not in config["variants"] or config["variants"][name]["route"] != route:
                        raise ValueError("A family component is undeclared or assigned to the wrong route.")
    if len(config["candidate_families"])*len(config["blend_weights_new_candidate"]) != 36:
        raise ValueError("This declaration requires exactly 36 candidate recipes.")
    if (len(set(config["blend_weights_new_candidate"])) != len(config["blend_weights_new_candidate"])
            or any(not np.isfinite(w) or not 0 < w <= 1 for w in config["blend_weights_new_candidate"])
            or not 0 < config["minimum_improvement_fraction_each_fold"] < 1):
        raise ValueError("Invalid convex weights or promotion margin.")


def reference_selection(root, config):
    path = root/config["reference_selection"]
    if sha256_file(path) != config["reference_selection_sha256"]:
        raise ValueError("Frozen v6 selection changed.")
    selection = read_json(path)
    if selection["audit_used"] or selection["selected_candidate"] != "cat_residual_v5_100":
        raise ValueError("Expected the frozen v6 recipe.")
    return selection


def aligned_predictions(features, saved):
    if (len(features) != len(saved) or saved[ID].duplicated().any()
            or identity_hash(features[ID]) != identity_hash(saved[ID])):
        raise ValueError("Predictions must cover the exact complete feature population.")
    result = features[[ID]].merge(saved[[ID, "prediction"]], on=ID, how="left", sort=False, validate="one_to_one")
    if not np.isfinite(result["prediction"]).all():
        raise ValueError("Every scoring row must have a finite prediction.")
    return result


def prepare_references(root, output, config, fold, features, truth, original, previous_runs):
    directory = output/"references"/fold
    selection = reference_selection(root, config)
    card = selection["development_runs"][config["reference_variant"]][fold]
    source = previous_runs/card["run_id"]
    if sha256_file(source/"run.json") != card["run_sha256"]:
        raise ValueError("The frozen v6 development run changed.")
    source_run = read_json(source/"run.json")
    if source_run["model_code"]["sha256"] != parent_code(root)["sha256"]:
        raise ValueError("Frozen v6 source code changed.")
    verify_receipts(source, source_run["artifacts"])
    reference = aligned_predictions(features, pd.read_parquet(source/"predictions.parquet"))
    metric = evaluate(truth, reference, original["ranking_month_weights"] if fold == "seasonal" else None)
    expected = selection["candidates"][selection["selected_candidate"]]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
    if not np.isclose(metric["overall"]["rmse_seconds"], expected, rtol=1e-12, atol=1e-9):
        raise ValueError("Frozen v6 complete score no longer reproduces.")
    if (directory/"reference.json").exists():
        report = read_json(directory/"reference.json")
        if (report["config_sha256"] != object_hash(config) or report["model_code_sha256"] != model_code_receipt(root)["sha256"]
                or report["source_run_sha256"] != card["run_sha256"]):
            raise ValueError("Saved ladder reference identity changed.")
        verify_receipts(directory, report["artifacts"])
        pd.testing.assert_frame_equal(pd.read_parquet(directory/"v6.parquet"), reference, check_exact=True)
        return reference, report
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Incomplete reference extraction exists; inspect it before resuming.")
    directory.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(directory/"v6.parquet", index=False, compression="zstd")
    nodes = read_json(source/"nodes.json")
    mask = route_mask(features, "clock_available")
    routed = features.loc[mask].reset_index(drop=True)
    global_values = old_node_prediction(source, nodes["global"], routed, config["threads"])
    airport_values = old_pipeline_prediction(source, {"kind": "nmclock_airports", "nodes": nodes["airports"]}, routed, config["threads"])
    delta = float(np.max(np.abs(.5*global_values+.5*airport_values-reference.loc[mask, "prediction"].to_numpy())))
    if delta > 1e-8:
        raise ValueError("The original v6 component weights do not reproduce its predictions.")
    for name, replacement in [("v6_global", global_values), ("v6_airports", airport_values)]:
        complete = reference.copy()
        complete.loc[mask, "prediction"] = replacement
        complete.to_parquet(directory/(name+".parquet"), index=False, compression="zstd")
    report = {"created_at_utc": utc_now(), "fold": fold, "config_sha256": object_hash(config),
              "model_code_sha256": model_code_receipt(root)["sha256"],
              "source_run": source.name, "source_run_sha256": card["run_sha256"],
              "source_predictions_sha256": sha256_file(source/"predictions.parquet"),
              "rows": len(features), "score_ids_sha256": identity_hash(features[ID]),
              "maximum_v6_reconstruction_difference_seconds": delta, "metrics": metric,
              "audit_used": False, "artifacts": receipts(directory)}
    write_json(directory/"reference.json", report)
    return reference, report


def checked_runs(root, output, config, manifest):
    result = {}
    code = model_code_receipt(root)["sha256"]
    for path in sorted(output.glob("*/run.json")):
        report = read_json(path)
        if report["stage"] != "development":
            continue
        if (report["config_sha256"] != object_hash(config) or report["bundle_id"] != manifest["bundle_id"]
                or report["model_code"]["sha256"] != code or report["audit_used"]):
            raise ValueError("Existing ladder run has different code, configuration or features.")
        key = (report["variant"], report["fold"])
        if key in result:
            raise ValueError("Duplicate completed ladder development run.")
        verify_receipts(path.parent, report["artifacts"])
        result[key] = (path.parent, report)
    return result


def fit_variant(run, name, variant, features, truth, stop=None, stop_truth=None, iterations=None):
    route = variant["route"]
    if not len(features) or not route_mask(features, route).all():
        raise ValueError("Fitting rows must belong to the declared nonempty route.")
    if stop is not None and (not len(stop) or not route_mask(stop, route).all()):
        raise ValueError("Stopping rows must belong to the same declared route.")
    function = fit_clock if variant["schema"] == "nmclock" else fit_weather
    return function(run, name, native_spec(variant), variant["settings"], features, truth, stop, stop_truth, iterations)


def suite(root, base, config_path, output, fold, previous_runs, variants=None):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    if config["feature_bundle_id"] != manifest["bundle_id"]:
        raise ValueError("The existing NM feature bundle identity changed.")
    existing = checked_runs(root, output, config, manifest)
    chosen = config["variant_order"] if variants is None else variants
    if set(chosen)-set(config["variants"]):
        raise ValueError("An undeclared variant was requested.")
    assignment = pd.read_parquet(parent/"splits"/f"{fold}.parquet")
    features, truths = {}, {}
    for role in ["fit", "early_stop", "score"]:
        features[role], truth = load_partition(base, parent, catalog, assignment, role, FEATURE_NAMES)
        truths[role] = aligned_truth(features[role], truth)
        if identity_hash(features[role][ID]) != original["folds"][fold]["identity_sha256"][role]:
            raise ValueError("Protected fitting, stopping or scoring identities changed.")
    reference, reference_receipt = prepare_references(
        root, output, config, fold, features["score"], truths["score"], original, previous_runs)
    for variant_name in chosen:
        if (variant_name, fold) in existing:
            emit("ladder_run_reused", variant=variant_name, fold=fold)
            continue
        if shutil.disk_usage(root).free < 1_000_000_000:
            raise ValueError("Less than 1 GB free: preserve existing artifacts before another fit.")
        variant = config["variants"][variant_name]
        masks = {role: route_mask(frame, variant["route"]) for role, frame in features.items()}
        routed = {role: frame.loc[masks[role]].reset_index(drop=True) for role, frame in features.items()}
        routed_truth = {role: truths[role].loc[masks[role]].reset_index(drop=True) for role in features}
        run = new_run(output, f"{fold}_{variant_name}")
        with ResourceMonitor() as monitor:
            node = fit_variant(run, "component", variant, routed["fit"], routed_truth["fit"],
                               routed["early_stop"], routed_truth["early_stop"])
            replacement = node_prediction(run, node, routed["score"], config["threads"])
            predictions = reference.copy()
            predictions.loc[masks["score"], "prediction"] = replacement
            predictions.to_parquet(run/"predictions.parquet", index=False, compression="zstd")
            pd.DataFrame({ID: routed["score"][ID], "prediction": replacement}).to_parquet(
                run/"routed_predictions.parquet", index=False, compression="zstd")
            metrics = evaluate(truths["score"], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
            for name, value in [("node", node), ("metrics", metrics), ("config", config), ("reference", reference_receipt)]:
                write_json(run/(name+".json"), value)
        report = {"created_at_utc": utc_now(), "run_id": run.name, "stage": "development",
                  "variant": variant_name, "fold": fold, "model_kind": variant["kind"], "route": variant["route"],
                  "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
                  "model_code": model_code_receipt(root), "config_sha256": object_hash(config),
                  "best_iterations": node["iterations"], "newly_fitted_models": 1,
                  "audit_used": False, "target_filter": "none", "prediction_postprocess": "none",
                  "reference_run_sha256": reference_receipt["source_run_sha256"],
                  "max_model_reload_difference_seconds": node["max_model_reload_difference_seconds"],
                  "environment": model_environment(root), "resources": monitor.stats}
        for role in features:
            report[role+"_rows"] = len(features[role])
            report[role+"_ids_sha256"] = identity_hash(features[role][ID])
            report["routed_"+role+"_rows"] = len(routed[role])
            report["routed_"+role+"_ids_sha256"] = identity_hash(routed[role][ID])
        report["artifacts"] = receipts(run)
        write_json(run/"run.json", report)
        existing[(variant_name, fold)] = (run, report)
        emit("ladder_run_complete", variant=variant_name, fold=fold,
             metrics=metrics["overall"], resources=monitor.stats)
        del routed, routed_truth, masks
        gc.collect()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/nmclock/72e86725e23367ede5b5"))
    parser.add_argument("--config", type=Path, default=Path("configs/ladder_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/ladder"))
    parser.add_argument("--previous-runs", type=Path, default=Path("runs/nmclock"))
    parser.add_argument("--fold", required=True, choices=["seasonal", "forward"])
    parser.add_argument("--variant", action="append")
    args = parser.parse_args()
    suite(root, root/args.bundle, root/args.config, root/args.output, args.fold, root/args.previous_runs, args.variant)
