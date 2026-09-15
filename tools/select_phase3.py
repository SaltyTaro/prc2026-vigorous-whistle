"""Rescore feature ablations against v1, report uncertainty, and freeze a recipe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.phase1 import load_partition as load_original_partition
from prc2026.validation import evaluate, identity_hash
from prc2026_models.experiments import model_code_receipt as original_model_code
from prc2026_phase3.bundle import load_bundle
from prc2026_phase3.training import model_code_receipt, validate_config


def paired_days(truth: pd.DataFrame, candidate: pd.DataFrame, reference: pd.DataFrame, seed: int) -> dict:
    merged = truth[[ID, TIME, TARGET]].merge(candidate, on=ID, validate="one_to_one").rename(columns={"prediction": "candidate"})
    merged = merged.merge(reference, on=ID, validate="one_to_one")
    merged["candidate_sse"] = (merged["candidate"]-merged[TARGET])**2
    merged["reference_sse"] = (merged["prediction"]-merged[TARGET])**2
    grouped = merged.groupby(merged[TIME].dt.strftime("%Y-%m-%d")).agg(
        candidate_sse=("candidate_sse", "sum"), reference_sse=("reference_sse", "sum"), rows=(ID, "count"))
    indexes = np.random.default_rng(seed).integers(0, len(grouped), size=(2000, len(grouped)))
    rows = grouped["rows"].to_numpy()[indexes].sum(axis=1)
    delta = (np.sqrt(grouped["candidate_sse"].to_numpy()[indexes].sum(axis=1)/rows)
             - np.sqrt(grouped["reference_sse"].to_numpy()[indexes].sum(axis=1)/rows))
    return {"comparison": "candidate minus phase-2 LightGBM", "days": len(grouped),
            "replicates": 2000, "rmse_difference_95pct_interval_seconds": list(map(float, np.quantile(delta, [.025, .975]))),
            "days_with_lower_squared_error": int((grouped["candidate_sse"] < grouped["reference_sse"]).sum()),
            "interpretation": "Descriptive uncertainty on reused development data, not an independent final test"}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/phase3"))
    parser.add_argument("--reference-runs", type=Path, default=Path("runs/phase2"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_v1.json"))
    parser.add_argument("--additional-config", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("runs/phase3/selection.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/phase3"))
    args = parser.parse_args()
    config = read_json(root / args.config)
    configurations = [(args.config, config)] + [(p, read_json(root / p)) for p in args.additional_config]
    base = root / (args.bundle or read_json(root / "data/phase3/latest.json")["bundle"])
    bundle, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    config_hash, code_hash = object_hash(config), model_code_receipt(root)["sha256"]
    variant_configs, examined_configs = {}, {}
    common_keys = set(config) - {"experiment_id", "variants"}
    for path, recipe in configurations:
        validate_config(recipe, original["descriptor"]["protocol"])
        if any(recipe[k] != config[k] for k in common_keys):
            raise ValueError("Follow-up configurations must preserve model/selection settings.")
        digest = object_hash(recipe)
        examined_configs[digest] = {"path": path.as_posix(), "configuration": recipe}
        for variant in recipe["variants"]:
            if variant in variant_configs:
                raise ValueError("Feature variants must have unique names across configurations.")
            variant_configs[variant] = (digest, path.as_posix(), recipe)
    baseline_config_hash = object_hash(read_json(root / "configs/baselines_v1.json"))
    baseline_code = original_model_code(root)["sha256"]
    references, experiments = {}, {}
    for filename in sorted((root / args.reference_runs).glob("*/run.json")):
        run = read_json(filename)
        if run.get("stage") != "development" or run["model_kind"] != "lightgbm":
            continue
        if run["config_sha256"] != baseline_config_hash:
            continue
        if run["bundle_id"] != original["bundle_id"] or run["model_code"]["sha256"] != baseline_code or run["audit_used"]:
            raise ValueError("Reference run provenance mismatch.")
        if run["fold"] in references:
            raise ValueError("Ambiguous duplicate reference runs.")
        verify_receipts(filename.parent, run["artifacts"])
        references[run["fold"]] = (filename.parent, run)
    if set(references) != set(config["development_folds"]):
        raise ValueError("Both frozen phase-2 reference predictions are required.")
    for filename in sorted((root / args.runs).glob("*/run.json")):
        run = read_json(filename)
        if run.get("stage") != "development" or run["config_sha256"] not in examined_configs:
            continue
        if run["audit_used"] or run["fold"] not in config["development_folds"]:
            raise ValueError("Audit data in feature selection.")
        if run["bundle_id"] != bundle["bundle_id"] or run["model_code"]["sha256"] != code_hash:
            raise ValueError("Experiment bundle/training code mismatch.")
        key = (run["variant"], run["fold"])
        if run["variant"] not in variant_configs or run["config_sha256"] != variant_configs[run["variant"]][0]:
            raise ValueError("Variant does not belong to the declared configuration.")
        if key in experiments:
            raise ValueError("Ambiguous duplicate feature experiments.")
        verify_receipts(filename.parent, run["artifacts"])
        experiments[key] = (filename.parent, run)
    expected = {(v, f) for v in variant_configs for f in config["development_folds"]}
    if set(experiments) != expected:
        raise ValueError("Every declared feature combination must be tested on both folds.")
    truths, reference_predictions, reference_reports = {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent / "splits" / f"{fold}.parquet")
        _, truth = load_original_partition(parent, catalog, assignment, "score")
        path, run = references[fold]
        predictions = pd.read_parquet(path / "predictions.parquet")
        metrics = evaluate(truth, predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
        if not np.isclose(metrics["overall"]["rmse_seconds"], read_json(path / "metrics.json")["overall"]["rmse_seconds"], rtol=1e-12):
            raise ValueError("Phase-2 reference does not rescore exactly.")
        truths[fold], reference_predictions[fold] = truth, predictions
        reference_reports[fold] = {"run_id": run["run_id"], "run_sha256": sha256_file(path / "run.json"), "metrics": metrics}
    candidates, airport_rows = {}, []
    for variant, (variant_hash, variant_config_path, variant_config) in variant_configs.items():
        folds, ratios = {}, []
        for fold in config["development_folds"]:
            path, run = experiments[(variant, fold)]
            predictions = pd.read_parquet(path / "predictions.parquet")
            if identity_hash(predictions[ID]) != original["folds"][fold]["identity_sha256"]["score"]:
                raise ValueError("Candidate scoring IDs changed.")
            metrics = evaluate(truths[fold], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
            if not np.isclose(metrics["overall"]["rmse_seconds"], read_json(path / "metrics.json")["overall"]["rmse_seconds"], rtol=1e-12):
                raise ValueError("Candidate predictions do not reproduce saved metrics.")
            reference_rmse = reference_reports[fold]["metrics"]["overall"]["rmse_seconds"]
            ratio = metrics["overall"]["rmse_seconds"]/reference_rmse
            ratios.append(ratio)
            folds[fold] = {"run_id": run["run_id"], "run_sha256": sha256_file(path / "run.json"),
                           "best_iterations": run["best_iterations"], "early_stop_rmse_seconds": run["early_stop_rmse_seconds"],
                           "metrics": metrics, "rmse_ratio_to_phase2_lightgbm": ratio,
                           "rmse_improvement_seconds": reference_rmse-metrics["overall"]["rmse_seconds"],
                           "paired_day_uncertainty": paired_days(truths[fold], predictions, reference_predictions[fold], config["seed"]),
                           "resources": run["resources"], "feature_count": len(run["feature_names"])}
            for airport, values in metrics["by_airport"].items():
                airport_rows.append({"variant": variant, "fold": fold, "airport": airport, **values})
        candidates[variant] = {"families": variant_config["variants"][variant], "folds": folds,
                               "config_sha256": variant_hash, "config_path": variant_config_path,
                               "improves_both_folds": all(r < 1 for r in ratios), "mean_relative_rmse": float(np.mean(ratios))}
    eligible = [v for v, c in candidates.items() if c["improves_both_folds"]]
    chosen = min(eligible, key=lambda v: candidates[v]["mean_relative_rmse"]) if eligible else None
    iterations = int(np.rint(np.median([f["best_iterations"] for f in candidates[chosen]["folds"].values()]))) if chosen else None
    selection = {
        "created_at_utc": utc_now(), "selected_model": "lightgbm", "selected_variant": chosen,
        "refit_iterations": iterations, "refit_months": config["refit_months"], "audit_used": False,
        "rule": config["selection"], "config_sha256": variant_configs[chosen][0] if chosen else config_hash,
        "config_path": variant_configs[chosen][1] if chosen else args.config.as_posix(),
        "examined_configurations": examined_configs, "model_code_sha256": code_hash,
        "selection_code_sha256": sha256_file(Path(__file__)), "bundle_id": bundle["bundle_id"],
        "parent_bundle_id": original["bundle_id"], "references": reference_reports, "candidates": candidates,
    }
    write_json(root / args.output, selection)
    report_dir = root / args.report_dir
    write_json(report_dir / "development_results.json", selection)
    write_json(report_dir / "feature_manifest.json", bundle)
    pd.DataFrame(airport_rows).to_csv(report_dir / "airport_errors.csv", index=False)
    print("Selected:", chosen, "refit iterations:", iterations)
    for variant, details in candidates.items():
        print(variant, {f: round(d["metrics"]["overall"]["rmse_seconds"], 6) for f, d in details["folds"].items()},
              "improves_both:", details["improves_both_folds"])


if __name__ == "__main__":
    main()
