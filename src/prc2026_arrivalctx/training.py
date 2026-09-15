"""Two declared missing-clock arrival-context learners on each complete fold."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.phase1 import emit
from prc2026.validation import evaluate, identity_hash, require_development_fold
from prc2026_models.experiments import model_environment, new_run, receipts
from prc2026_phase5r.baseline import aligned_truth
from .inference import route_mask
from prc2026_ladder.training import model_code_receipt as parent_code, prepare_references
from .bundle import feature_code_receipt, load_bundle, load_partition
from .features import FEATURE_NAMES, MODEL_NAMES
from .encoding import contract
from .inference import node_prediction
from .native import fit_node


def model_code_receipt(root):
    hashes = dict(parent_code(root)["files"])
    hashes.update(feature_code_receipt(root)["files"])
    paths = [root/"src/prc2026_arrivalctx"/name for name in ["encoding.py", "native.py", "inference.py", "training.py"]]
    paths += [root/"tools/run_arrivalctx.py"]
    hashes.update({p.relative_to(root).as_posix(): sha256_file(p) for p in paths})
    return {"files": hashes, "sha256": object_hash(hashes)}


def validate_config(root, config, protocol):
    if config["validation_protocol"] != protocol["protocol_id"] or config["development_folds"] != ["seasonal", "forward"]:
        raise ValueError("Both original complete development folds are required.")
    if config["audit_used"] or config["target_filter"] != "none" or config["prediction_postprocess"] != "none":
        raise ValueError("No audit, target filtering or clipping is allowed.")
    if config["refit_months"] != [f"2025-{m:02}" for m in range(1, 12)] or config["threads"] != 16 or config["seed"] != 20260913:
        raise ValueError("Protected refit or fixed compute settings changed.")
    if config["feature_names"] != MODEL_NAMES or config["variant_order"] != ["lgb_arrival", "cat_arrival"] or set(config["variants"]) != {"lgb_arrival", "cat_arrival"}:
        raise ValueError("The two feature comparisons changed.")
    prior_path = root / config["matched_reference_selection"]
    if sha256_file(prior_path) != config["matched_reference_selection_sha256"]:
        raise ValueError("Matched baseline evidence changed.")
    prior = read_json(prior_path)["configuration"]["variants"]
    for name, previous in [("lgb_arrival", "lgb_missing_regularized"), ("cat_arrival", "cat_missing_ground")]:
        variant = config["variants"][name]
        expected = {"kind": prior[previous]["kind"], "features": "ground_weather_arrival", "route": "clock_missing",
                    "target": "conditional_schedule_residual", "settings": prior[previous]["settings"], "matched_reference_variant": previous}
        if variant != expected:
            raise ValueError("A learner differs from its matched declared settings.")
        contract(native_spec(variant))
    families = {name: {"clock_available": {"v6": 1.0}, "clock_missing": {name: 1.0}} for name in config["variants"]}
    families["arrival_mean"] = {"clock_available": {"v6": 1.0}, "clock_missing": {"lgb_arrival": .5, "cat_arrival": .5}}
    if config["candidate_families"] != families or config["blend_weights_new_candidate"] != [.5, 1.] or config["minimum_improvement_fraction_each_fold"] != .005:
        raise ValueError("The fixed six recipes or promotion margin changed.")
    for fold in config["development_folds"]:
        require_development_fold(protocol, fold)


def native_spec(variant):
    return {key: variant[key] for key in ["kind", "features", "route", "target"]}


def checked_runs(root, output, config, manifest):
    found = {}
    code = model_code_receipt(root)
    for path in sorted(output.glob("*/run.json")):
        report = read_json(path)
        if report["stage"] != "development":
            continue
        if report["model_code"] != code or report["config_sha256"] != object_hash(config) or report["bundle_id"] != manifest["bundle_id"] or report["audit_used"]:
            raise ValueError("A saved arrivalctx run has different scientific inputs.")
        key = (report["variant"], report["fold"])
        if key in found:
            raise ValueError("Duplicate development run.")
        verify_receipts(path.parent, report["artifacts"])
        found[key] = (path.parent, report)
    return found


def suite(root, base, config_path, output, fold, previous_runs):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(root, config, original["descriptor"]["protocol"])
    if sha256_file(root/"configs/features_arrivalctx_v1.json") != config["feature_configuration_sha256"]:
        raise ValueError("Feature declaration changed.")
    if manifest["descriptor"]["config"] != read_json(root/"configs/features_arrivalctx_v1.json"):
        raise ValueError("Model and feature declarations disagree.")
    existing = checked_runs(root, output, config, manifest)
    assignment = pd.read_parquet(parent/"splits"/f"{fold}.parquet")
    features, truths, routed, routed_truth, masks = {}, {}, {}, {}, {}
    for role in ["fit", "early_stop", "score"]:
        features[role], truth = load_partition(base, parent, catalog, assignment, role, FEATURE_NAMES)
        truths[role] = aligned_truth(features[role], truth)
        if identity_hash(features[role][ID]) != original["folds"][fold]["identity_sha256"][role]:
            raise ValueError("Protected split populations changed.")
        masks[role] = route_mask(features[role])
        routed[role] = features[role].loc[masks[role]].reset_index(drop=True)
        routed_truth[role] = truths[role].loc[masks[role]].reset_index(drop=True)
    reference, ref = prepare_references(root, output, config, fold, features["score"], truths["score"], original, previous_runs)
    for name in config["variant_order"]:
        if (name, fold) in existing:
            continue
        if shutil.disk_usage(root).free < 900_000_000:
            raise ValueError("Less than 900 MB free before another fit.")
        spec = native_spec(config["variants"][name])
        run = new_run(output, f"{fold}_{name}")
        settings = config["variants"][name]["settings"]
        with ResourceMonitor() as monitor:
            node = fit_node(run, "component", spec, settings, routed["fit"], routed_truth["fit"], routed["early_stop"], routed_truth["early_stop"])
            replacement = node_prediction(run, node, routed["score"], config["threads"])
            predictions = reference.copy()
            predictions.loc[masks["score"], "prediction"] = replacement
            predictions.to_parquet(run/"predictions.parquet", index=False, compression="zstd")
            pd.DataFrame({ID: routed["score"][ID], "prediction": replacement}).to_parquet(run/"routed_predictions.parquet", index=False, compression="zstd")
            metrics = evaluate(truths["score"], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
            for filename, value in [("node", node), ("config", config), ("metrics", metrics), ("reference", ref)]:
                write_json(run/(filename+".json"), value)
        report = {"created_at_utc": utc_now(), "run_id": run.name, "stage": "development", "variant": name, "fold": fold,
                  "model_kind": spec["kind"], "route": "clock_missing", "bundle_id": manifest["bundle_id"],
                  "parent_bundle_id": original["bundle_id"], "model_code": model_code_receipt(root), "config_sha256": object_hash(config),
                  "best_iterations": node["iterations"], "newly_fitted_models": 1, "audit_used": False,
                  "target_filter": "none", "prediction_postprocess": "none", "reference_run_sha256": ref["source_run_sha256"],
                  "max_model_reload_difference_seconds": node["max_model_reload_difference_seconds"],
                  "environment": model_environment(root), "resources": monitor.stats}
        for role in features:
            report[role+"_rows"] = len(features[role])
            report[role+"_ids_sha256"] = identity_hash(features[role][ID])
            report["routed_"+role+"_rows"] = len(routed[role])
            report["routed_"+role+"_ids_sha256"] = identity_hash(routed[role][ID])
        report["artifacts"] = receipts(run)
        write_json(run/"run.json", report)
        emit("arrivalctx_native_complete", variant=name, fold=fold, complete_component_metrics=metrics["overall"], resources=monitor.stats)
        gc.collect()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/arrivalctx_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/arrivalctx"))
    parser.add_argument("--previous-runs", type=Path, default=Path("runs/nmclock"))
    parser.add_argument("--fold", required=True, choices=["seasonal", "forward"])
    args = parser.parse_args()
    base = args.bundle or Path(read_json(root/"data/arrivalctx/latest.json")["bundle"])
    suite(root, root/base, root/args.config, root/args.output, args.fold, root/args.previous_runs)
