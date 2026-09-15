"""Refit a verified ladder recipe with protected December flight groups."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TIME
from prc2026.phase1 import emit
from prc2026.validation import identity_hash
from prc2026_models.experiments import model_environment, new_run, receipts
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_nmclock.bundle import load_bundle
from prc2026_nmclock.features import FEATURE_NAMES
from prc2026_nmclock.inference import copy_file_tree
from prc2026_ladder.inference import pipeline_prediction, recipe_pipeline, route_mask
from prc2026_ladder.training import checked_runs, fit_variant, model_code_receipt, validate_config
from select_ladder import active_variants, eligible_recipe


def refit_assignment(catalog, protocol, months):
    movement_months = catalog[TIME].dt.strftime("%Y-%m")
    protected = set(catalog.loc[movement_months.isin(protocol["reserved_audit_months"]), "FLIGHT_ID_mvt"].dropna())
    assignment = catalog[[ID]].copy()
    assignment["partition"] = np.where(
        movement_months.isin(months) & ~catalog["FLIGHT_ID_mvt"].isin(protected), "fit", "reserved")
    return assignment


def copy_reference(run, source):
    nodes = read_json(source/"nodes.json")["cat_residual"]
    copied = copy_file_tree(run, source, {
        "base": read_json(source/"pipeline.json"), "global": nodes["global"],
        "airports": {"kind": "nmclock_airports", "nodes": nodes["airports"]},
    }, "v6_")
    return {
        "v6": {"kind": "frozen_nmclock", "pipeline": copied["base"]},
        "v6_global": {"kind": "node", "node": copied["global"]},
        "v6_airports": {"kind": "frozen_nmclock", "pipeline": copied["airports"]},
    }


def require_verifications(root, selection_path, selection, code):
    reports = {}
    for name in ["verification.json", "local_verification.json"]:
        path = root/"docs/ladder"/name
        report = read_json(path)
        if (not report["passed"] or report["selection_sha256"] != sha256_file(selection_path)
                or report["model_code_sha256"] != code["sha256"]
                or report["selected_candidate"] != selection["selected_candidate"]
                or report["bundle_id"] != selection["bundle_id"]):
            raise ValueError("Matching Azure and Windows development verifications are required.")
        reports[name] = sha256_file(path)
    return reports


def refit(root, base, config_path, output, selection_path, previous_runs):
    config, selection = read_json(config_path), read_json(selection_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    code = model_code_receipt(root)
    candidate = eligible_recipe(config, selection, manifest, code)
    if selection["selection_code_sha256"] != sha256_file(root/"tools/select_ladder.py"):
        raise ValueError("The selection implementation changed.")
    verification_receipts = require_verifications(root, selection_path, selection, code)
    experiments = checked_runs(root, output, config, manifest)
    if set(experiments) != {(name, fold) for name in config["variant_order"] for fold in config["development_folds"]}:
        raise ValueError("The entire declared batch must finish before refit.")
    for (name, fold), (directory, _) in experiments.items():
        saved = selection["development_runs"][name][fold]
        if directory.name != saved["run_id"] or sha256_file(directory/"run.json") != saved["run_sha256"]:
            raise ValueError("Selected development run identity changed.")
    source = previous_runs/config["reference_refit_run"]
    if sha256_file(source/"run.json") != config["reference_refit_sha256"]:
        raise ValueError("Frozen v6 refit changed.")
    source_run = read_json(source/"run.json")
    verify_receipts(source, source_run["artifacts"])
    if any(read_json(path)["stage"] == "refit" for path in output.glob("*/run.json")):
        raise ValueError("A completed ladder refit already exists; preserve it.")
    run = new_run(output, "refit_"+selection["selected_candidate"])
    with ResourceMonitor() as monitor:
        assignment = refit_assignment(catalog, original["descriptor"]["protocol"], config["refit_months"])
        features, truth = load_partition(base, parent, catalog, assignment, "fit", FEATURE_NAMES)
        truth = aligned_truth(features, truth)
        if len(features) != source_run["fit_rows"] or identity_hash(features[ID]) != source_run["fit_ids_sha256"]:
            raise ValueError("The January-November refit population differs from protected v6.")
        sources = copy_reference(run, source)
        cards = {}
        for name in active_variants(config, candidate):
            variant = config["variants"][name]
            mask = route_mask(features, variant["route"])
            cards[name] = fit_variant(
                run, name, variant, features.loc[mask].reset_index(drop=True),
                truth.loc[mask].reset_index(drop=True), iterations=selection["refit_iterations"][name])
            sources[name] = {"kind": "node", "node": cards[name]}
        pipeline = recipe_pipeline(candidate["routes"], candidate["new_weight"], sources)
        for name, value in [("pipeline", pipeline), ("nodes", cards), ("sources", sources),
                            ("selection", selection), ("config", config)]:
            write_json(run/(name+".json"), value)
        ranking = pd.read_parquet(base/"features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking IDs changed.")
        values = pipeline_prediction(run, pipeline, ranking, config["threads"])
        predictions = pd.DataFrame({ID: ranking[ID], "prediction": values})
        predictions.to_parquet(run/"ranking_predictions.parquet", index=False, compression="zstd")
        indexes = np.unique(np.r_[np.linspace(0, len(ranking)-1, 1000, dtype=int),
                                 np.flatnonzero(route_mask(ranking, "clock_missing")),
                                 np.flatnonzero(ranking["lirf_missing_record"].eq(1)),
                                 np.flatnonzero(ranking["nmclock_proxy_s"].gt(10800) | ranking["nmclock_proxy_s"].lt(0))])
        restored = pipeline_prediction(run, read_json(run/"pipeline.json"), ranking.iloc[indexes].reset_index(drop=True), config["threads"])
        delta = float(np.max(np.abs(values[indexes]-restored)))
        if delta > 1e-8:
            raise ValueError("The saved complete ladder pipeline does not reproduce.")
    report = {
        "created_at_utc": utc_now(), "run_id": run.name, "stage": "refit", "model_kind": "ensemble",
        "variant": selection["selected_candidate"], "iterations": sum(selection["refit_iterations"].values()),
        "iterations_scope": "New native trees; imported v6 models retain their original counts.",
        "component_iterations": selection["refit_iterations"], "newly_fitted_models": len(cards),
        "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
        "model_code": code, "refit_code_sha256": sha256_file(Path(__file__)),
        "config_sha256": object_hash(config), "selection_sha256": sha256_file(selection_path),
        "source_refit_sha256": config["reference_refit_sha256"], "verification_receipts": verification_receipts,
        "fit_rows": len(features), "fit_ids_sha256": identity_hash(features[ID]),
        "refit_months": config["refit_months"], "audit_used": False, "target_filter": "none",
        "prediction_postprocess": "none", "ranking_rows": len(ranking),
        "ranking_ids_sha256": identity_hash(ranking[ID]),
        "reload_sample_rows": len(indexes), "max_model_reload_difference_seconds": delta,
        "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run),
    }
    write_json(run/"run.json", report)
    emit("ladder_refit_complete", run=str(run), ranking_rows=len(ranking), resources=monitor.stats)
    return run


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/nmclock/72e86725e23367ede5b5"))
    parser.add_argument("--config", type=Path, default=Path("configs/ladder_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/ladder"))
    parser.add_argument("--selection", type=Path, default=Path("runs/ladder/selection.json"))
    parser.add_argument("--previous-runs", type=Path, default=Path("runs/nmclock"))
    args = parser.parse_args()
    refit(root, root/args.bundle, root/args.config, root/args.runs, root/args.selection, root/args.previous_runs)


if __name__ == "__main__":
    main()
