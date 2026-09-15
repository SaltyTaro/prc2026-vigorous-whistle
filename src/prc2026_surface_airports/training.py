"""Fit every airport with unchanged settings and the frozen combined features."""
from pathlib import Path
from copy import deepcopy
import argparse
import gc
import shutil

import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TIMEZONES
from prc2026.phase1 import emit
from prc2026.validation import evaluate, identity_hash, require_development_fold
from prc2026_models.experiments import model_environment, new_run, receipts
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_nmclock.features import route_mask
from prc2026_ladder.training import prepare_references, reference_selection
from prc2026_timeline.bundle import load_bundle
from prc2026_timeline.features import FEATURE_NAMES
from prc2026_timeline.encoding import contract
from prc2026_timeline.native import fit_node
from prc2026_timeline.training import model_code_receipt as parent_code
from .inference import airport_prediction


def model_code_receipt(root):
    files = dict(parent_code(root)["files"])
    paths = list((root / "src/prc2026_surface_airports").glob("*.py"))
    paths += [root / "tools/run_surface_airports.py", root / "requirements-surface-airports.txt"]
    files.update({path.relative_to(root).as_posix(): sha256_file(path) for path in paths})
    return {"files": files, "sha256": object_hash(files)}


def validate_config(root, config, protocol):
    if config["audit_used"] or config["target_filter"] != "none" or config["prediction_postprocess"] != "none":
        raise ValueError("The reserved audit, label filtering and clipping remain prohibited.")
    if config["validation_protocol"] != protocol["protocol_id"] or config["development_folds"] != ["seasonal", "forward"]:
        raise ValueError("Both original complete folds are required.")
    if config["airports"] != sorted(TIMEZONES) or config["threads"] != 16 or config["seed"] != 20260913:
        raise ValueError("All ten airports and the original compute settings are required.")
    if config["refit_months"] != [f"2025-{month:02}" for month in range(1, 12)]:
        raise ValueError("December must remain outside refitting.")
    spec = {"kind": "lightgbm", "features": "combined", "target": "nm_residual", "route": "clock_available"}
    if config["native_spec"] != spec or config["feature_names"] != contract(spec)[0]:
        raise ValueError("The already frozen combined feature contract changed.")
    old = reference_selection(root, config)["configuration"]
    expected = deepcopy(old["lightgbm"])
    expected["parameters"].update(old["airport_lightgbm"]["parameters"])
    for key in ["rounds", "early_stopping_rounds"]:
        expected[key] = old["airport_lightgbm"][key]
    if config["settings"] != {"lightgbm": expected, "threads": old["threads"], "seed": old["seed"]}:
        raise ValueError("Airport learner settings differ from the original NM models.")
    families = {"airport_only": {"clock_available": {"v6_global": .5, "surface_airports": .5}, "clock_missing": {"v6": 1.}},
                "full_timeline": {"clock_available": {"lgb_combined": .25, "cat_combined": .25, "surface_airports": .5}, "clock_missing": {"v6": 1.}}}
    if config["candidate_families"] != families or config["blend_weights_new_candidate"] != [.5, 1.] or config["minimum_improvement_fraction_each_fold"] != .005:
        raise ValueError("The four fixed recipes or promotion margin changed.")
    if sha256_file(root / config["timeline_selection"]) != config["timeline_selection_sha256"]:
        raise ValueError("The prior complete timeline evidence changed.")
    for fold in config["development_folds"]:
        require_development_fold(protocol, fold)


def checked_runs(root, output, config, manifest):
    found = {}
    code = model_code_receipt(root)
    for path in sorted(output.glob("*/run.json")):
        report = read_json(path)
        if report["stage"] != "development":
            continue
        if report["config_sha256"] != object_hash(config) or report["model_code"] != code or report["bundle_id"] != manifest["bundle_id"] or report["audit_used"]:
            raise ValueError("An airport run has changed scientific inputs.")
        if report["fold"] in found or report["newly_fitted_models"] != 10 or report["variant"] != "surface_airports":
            raise ValueError("Each complete fold must contain exactly ten airport fits.")
        verify_receipts(path.parent, report["artifacts"])
        found[report["fold"]] = (path.parent, report)
    return found


def suite(root, base, config_path, output, fold, previous_runs):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(root, config, original["descriptor"]["protocol"])
    if manifest["bundle_id"] != config["feature_bundle_id"] or sha256_file(base / "manifest.json") != config["feature_manifest_sha256"]:
        raise ValueError("Frozen timeline feature bundle changed.")
    existing = checked_runs(root, output, config, manifest)
    if fold in existing:
        return existing[fold][0]
    assignment = pd.read_parquet(parent / "splits" / (fold + ".parquet"))
    features, truths, routed, routed_truth, masks = {}, {}, {}, {}, {}
    for role in ["fit", "early_stop", "score"]:
        features[role], truth = load_partition(base, parent, catalog, assignment, role, FEATURE_NAMES)
        truths[role] = aligned_truth(features[role], truth)
        if identity_hash(features[role][ID]) != original["folds"][fold]["identity_sha256"][role]:
            raise ValueError("Protected scoring or fitting identities changed.")
        masks[role] = route_mask(features[role])
        routed[role] = features[role].loc[masks[role]].reset_index(drop=True)
        routed_truth[role] = truths[role].loc[masks[role]].reset_index(drop=True)
    reference, ref = prepare_references(root, output, config, fold, features["score"], truths["score"], original, previous_runs)
    prior = reference_selection(root, config)
    old_dir = previous_runs / prior["development_runs"][config["reference_variant"]][fold]["run_id"]
    original_nodes = read_json(old_dir / "nodes.json")["airports"]
    if set(original_nodes) != set(config["airports"]) or any(node["settings"] != config["settings"] for node in original_nodes.values()):
        raise ValueError("The matched original airport settings differ.")
    run = new_run(output, fold + "_surface_airports")
    nodes = {}
    with ResourceMonitor() as monitor:
        for airport in config["airports"]:
            if shutil.disk_usage(root).free < 900_000_000:
                raise ValueError("Less than 900 MB free before another airport fit.")
            parts, labels = {}, {}
            for role in ["fit", "early_stop"]:
                mask = routed[role].airport.eq(airport)
                parts[role] = routed[role].loc[mask].reset_index(drop=True)
                labels[role] = routed_truth[role].loc[mask].reset_index(drop=True)
            nodes[airport] = fit_node(run, "airport_" + airport, config["native_spec"], config["settings"],
                                      parts["fit"], labels["fit"], parts["early_stop"], labels["early_stop"])
            write_json(run / "nodes_progress.json", nodes)
            emit("surface_airport_complete", fold=fold, airport=airport, iterations=nodes[airport]["iterations"], resources=nodes[airport]["resources"])
            gc.collect()
        values = airport_prediction(run, nodes, routed["score"], config["threads"])
        prediction = reference.copy()
        prediction.loc[masks["score"], "prediction"] = values
        prediction.to_parquet(run / "predictions.parquet", index=False, compression="zstd")
        metrics = evaluate(truths["score"], prediction, original["ranking_month_weights"] if fold == "seasonal" else None)
        for name, payload in [("nodes", nodes), ("config", config), ("metrics", metrics), ("reference", ref)]:
            write_json(run / (name + ".json"), payload)
    report = {"created_at_utc": utc_now(), "run_id": run.name, "stage": "development", "variant": "surface_airports", "fold": fold,
              "bundle_id": manifest["bundle_id"], "config_sha256": object_hash(config), "model_code": model_code_receipt(root),
              "newly_fitted_models": 10, "audit_used": False, "target_filter": "none", "prediction_postprocess": "none",
              "iterations": {name: node["iterations"] for name, node in nodes.items()}, "reference_run_sha256": ref["source_run_sha256"],
              "environment": model_environment(root), "resources": monitor.stats}
    for role in features:
        report[role + "_rows"] = len(features[role])
        report[role + "_ids_sha256"] = identity_hash(features[role][ID])
        report["routed_" + role + "_rows"] = len(routed[role])
        report["routed_" + role + "_ids_sha256"] = identity_hash(routed[role][ID])
    report["artifacts"] = receipts(run)
    write_json(run / "run.json", report)
    emit("surface_airports_fold_complete", fold=fold, component_metrics=metrics["overall"], resources=monitor.stats)
    return run


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/timeline/ce42240e5e43eee1713f"))
    parser.add_argument("--config", type=Path, default=Path("configs/surface_airports_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/surface_airports"))
    parser.add_argument("--previous-runs", type=Path, default=Path("runs/nmclock"))
    parser.add_argument("--fold", choices=["seasonal", "forward"], required=True)
    args = parser.parse_args()
    suite(root, root / args.bundle, root / args.config, root / args.output, args.fold, root / args.previous_runs)
