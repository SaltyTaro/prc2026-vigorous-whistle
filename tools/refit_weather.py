"""Refit only an unchanged, eligible weather recipe without inspecting December."""
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
from prc2026_weather.bundle import load_bundle
from prc2026_weather.features import FEATURE_NAMES
from prc2026_weather.inference import copy_legacy_pipeline, pipeline_prediction
from prc2026_weather.training import checked_runs, fit_node, model_code_receipt, settings_for, validate_config


def eligible_recipe(config, selection, manifest, code):
    if (selection["audit_used"] or selection["selected_candidate"] is None
        or selection["config_sha256"] != object_hash(config) or selection["bundle_id"] != manifest["bundle_id"]
        or selection["model_code_sha256"] != code["sha256"]):
        raise ValueError("An unchanged eligible selection is required before refit.")
    candidate = selection["candidates"][selection["selected_candidate"]]
    if not candidate["eligible"] or candidate["family_weights"] != config["candidate_families"][candidate["variant"]]:
        raise ValueError("The declared family was not eligible.")
    if candidate["new_weight"] not in config["blend_weights_new_candidate"]:
        raise ValueError("Undeclared final ensemble weight.")
    if set(candidate["folds"]) != set(config["development_folds"]):
        raise ValueError("Both complete validation folds are required.")
    for fold, row in candidate["folds"].items():
        reference = selection["reference_metrics"][fold]["overall"]["rmse_seconds"]
        ratio = row["metrics"]["overall"]["rmse_seconds"]/reference
        if not np.isfinite(ratio) or ratio > 1-config["minimum_improvement_fraction_each_fold"]:
            raise ValueError("The candidate fails the declared two-fold improvement margin.")
    for variant in candidate["family_weights"]:
        iterations = int(np.rint(np.median([selection["development_runs"][variant][fold]["iterations"] for fold in config["development_folds"]])))
        if selection["refit_iterations"].get(variant) != iterations or iterations < 1:
            raise ValueError("Fixed native iterations differ from October-selected medians.")
    return candidate


def refit(root, base, config_path, output, selection_path, prior):
    config, selection = read_json(config_path), read_json(selection_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    code = model_code_receipt(root)
    candidate = eligible_recipe(config, selection, manifest, code)
    experiments = checked_runs(root, output, config, manifest)
    expected = {(v, f) for v in config["variant_order"] for f in config["development_folds"]}
    if set(experiments) != expected:
        raise ValueError("All declared experiments must finish before a refit.")
    for (variant, fold), (path, report) in experiments.items():
        selected_run = selection["development_runs"][variant][fold]
        if selected_run["run_id"] != path.name or selected_run["run_sha256"] != sha256_file(path/"run.json"):
            raise ValueError("Selected development run identity changed.")
    source = prior/config["reference_refit_run"]
    if sha256_file(source/"run.json") != config["reference_refit_sha256"]:
        raise ValueError("Frozen v4 refit changed.")
    verify_receipts(source, read_json(source/"run.json")["artifacts"])
    if any(read_json(p)["stage"] == "refit" for p in output.glob("*/run.json")):
        raise ValueError("A completed weather refit already exists; preserve it.")
    run = new_run(output, "refit_"+selection["selected_candidate"])
    with ResourceMonitor() as monitor:
        months = catalog[TIME].dt.strftime("%Y-%m")
        protected = set(catalog.loc[months.isin(original["descriptor"]["protocol"]["reserved_audit_months"]), "FLIGHT_ID_mvt"].dropna())
        assignment = catalog[[ID]].copy()
        mask = months.isin(config["refit_months"]) & ~catalog["FLIGHT_ID_mvt"].isin(protected)
        assignment["partition"] = np.where(mask, "fit", "reserved")
        features, truth = load_partition(base, parent, catalog, assignment, "fit", FEATURE_NAMES)
        cards, members = {}, []
        for variant, weight in candidate["family_weights"].items():
            spec = config["variants"][variant]
            card = fit_node(run, variant, spec, settings_for(config, spec), features, truth,
                            iterations=selection["refit_iterations"][variant])
            cards[variant] = card
            members.append({"weight": weight, "pipeline": {"kind": "weather", "node": card}})
        old = copy_legacy_pipeline(run, source, "v4")
        original_airports = next(m["pipeline"] for m in old["members"] if m["name"] == "catboost_long")
        new = {"kind": "mixture", "members": [{"weight": .5, "pipeline": {"kind": "mixture", "members": members}},
                                                  {"weight": .5, "pipeline": {"kind": "airports", "pipeline": original_airports}}]}
        weight = candidate["new_weight"]
        pipeline = new if weight == 1 else {"kind": "mixture", "members": [
            {"weight": weight, "pipeline": new}, {"weight": 1-weight, "pipeline": {"kind": "legacy", "pipeline": old}}]}
        for name, value in [("pipeline", pipeline), ("nodes", cards), ("selection", selection), ("config", config)]:
            write_json(run/(name+".json"), value)
        ranking = pd.read_parquet(base/"features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking population changed.")
        values = pipeline_prediction(run, pipeline, ranking, config["threads"])
        predictions = pd.DataFrame({ID: ranking[ID], "prediction": values})
        predictions.to_parquet(run/"ranking_predictions.parquet", index=False, compression="zstd")
        indexes = np.unique(np.r_[np.linspace(0, len(ranking)-1, 1000, dtype=int),
                                  np.flatnonzero(ranking["lirf_missing_record"].eq(1)), np.flatnonzero(ranking["wx_missing"].eq(1))])
        restored = pipeline_prediction(run, read_json(run/"pipeline.json"), ranking.iloc[indexes].reset_index(drop=True), config["threads"])
        delta = float(np.max(np.abs(values[indexes]-restored)))
        if delta > 1e-8:
            raise ValueError("Saved final pipeline does not reproduce ranking predictions.")
    report = {"created_at_utc": utc_now(), "run_id": run.name, "stage": "refit", "model_kind": "ensemble",
              "variant": selection["selected_candidate"], "iterations": sum(selection["refit_iterations"].values()),
              "component_iterations": selection["refit_iterations"], "bundle_id": manifest["bundle_id"],
              "parent_bundle_id": original["bundle_id"], "feature_code": manifest["code"], "model_code": code,
              "refit_code_sha256": sha256_file(Path(__file__)), "config_sha256": object_hash(config),
              "selection_sha256": sha256_file(selection_path), "fit_rows": len(features), "fit_ids_sha256": identity_hash(features[ID]),
              "refit_months": config["refit_months"], "audit_used": False, "target_filter": "none", "prediction_postprocess": "none",
              "ranking_rows": len(ranking), "ranking_ids_sha256": identity_hash(ranking[ID]), "max_model_reload_difference_seconds": delta,
              "source_refit_sha256": config["reference_refit_sha256"], "environment": model_environment(root),
              "resources": monitor.stats, "artifacts": receipts(run)}
    write_json(run/"run.json", report)
    emit("weather_refit_complete", run=str(run), ranking_rows=len(ranking), resources=monitor.stats)
    return run


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/weather_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/weather"))
    parser.add_argument("--selection", type=Path, default=Path("runs/weather/selection.json"))
    parser.add_argument("--prior-runs", type=Path, default=Path("runs/phase5"))
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/"data/weather/latest.json")["bundle"])
    refit(root, base, root/args.config, root/args.runs, root/args.selection, root/args.prior_runs)


if __name__ == "__main__":
    main()
