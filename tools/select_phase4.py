"""Rescore every declared Phase-4 experiment and freeze a component-level refit."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.phase1 import load_partition as original_partition
from prc2026.validation import evaluate, identity_hash
from prc2026_phase4.bundle import load_bundle
from prc2026_phase4.training import model_code_receipt, reference_runs, validate_config


def paired_days(truth, candidate, reference, seed):
    merged = truth[[ID, TIME, TARGET]].merge(candidate, on=ID, validate="one_to_one").rename(columns={"prediction": "candidate"})
    merged = merged.merge(reference, on=ID, validate="one_to_one")
    merged["candidate_sse"] = (merged["candidate"]-merged[TARGET])**2
    merged["reference_sse"] = (merged["prediction"]-merged[TARGET])**2
    grouped = merged.groupby(merged[TIME].dt.strftime("%Y-%m-%d")).agg(
        candidate_sse=("candidate_sse", "sum"), reference_sse=("reference_sse", "sum"), rows=(ID, "count"))
    indices = np.random.default_rng(seed).integers(0, len(grouped), size=(2000, len(grouped)))
    rows = grouped["rows"].to_numpy()[indices].sum(axis=1)
    delta = (np.sqrt(grouped["candidate_sse"].to_numpy()[indices].sum(axis=1)/rows)
             - np.sqrt(grouped["reference_sse"].to_numpy()[indices].sum(axis=1)/rows))
    return {"comparison": "candidate minus Phase-3 timing_traffic", "days": len(grouped), "replicates": 2000,
            "rmse_difference_95pct_interval_seconds": list(map(float, np.quantile(delta, [.025, .975]))),
            "days_with_lower_squared_error": int((grouped["candidate_sse"] < grouped["reference_sse"]).sum()),
            "interpretation": "Descriptive on reused development data, not an independent audit"}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/phase4"))
    parser.add_argument("--reference-runs", type=Path, default=Path("runs/phase3"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4/selection.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/phase4"))
    args = parser.parse_args()
    config = read_json(root / args.config)
    base = root / (args.bundle or read_json(root / "data/phase4/latest.json")["bundle"])
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    code, config_hash = model_code_receipt(root), object_hash(config)
    references = reference_runs(root, root / args.reference_runs, config, original)
    experiments = {}
    for filename in sorted((root / args.runs).glob("*/run.json")):
        run = read_json(filename)
        if run["stage"] != "development":
            continue
        key = (run["variant"], run["fold"])
        if key in experiments or key[0] not in config["variants"] or key[1] not in config["development_folds"]:
            raise ValueError("Ambiguous or undeclared experiment.")
        if (run["config_sha256"] != config_hash or run["model_code"]["sha256"] != code["sha256"]
                or run["bundle_id"] != manifest["bundle_id"] or run["audit_used"]):
            raise ValueError("Experiment provenance mismatch.")
        for role in ["fit", "early_stop", "score"]:
            if run[role + "_ids_sha256"] != original["folds"][key[1]]["identity_sha256"][role]:
                raise ValueError("Protected experiment IDs changed.")
        verify_receipts(filename.parent, run["artifacts"])
        experiments[key] = (filename.parent, run)
    expected = {(v, f) for v in config["variants"] for f in config["development_folds"]}
    if set(experiments) != expected:
        raise ValueError("All declared experiments must finish before selection.")
    truths, reference_predictions, reference_reports = {}, {}, {}
    for fold, (path, run) in references.items():
        assignment = pd.read_parquet(parent / "splits" / f"{fold}.parquet")
        _, truth = original_partition(parent, catalog, assignment, "score")
        predictions = pd.read_parquet(path / "predictions.parquet")
        metrics = evaluate(truth, predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
        if not np.isclose(metrics["overall"]["rmse_seconds"], read_json(path / "metrics.json")["overall"]["rmse_seconds"], rtol=1e-12):
            raise ValueError("Reference predictions do not reproduce their RMSE.")
        truths[fold], reference_predictions[fold] = truth, predictions
        reference_reports[fold] = {"run_id": run["run_id"], "run_sha256": sha256_file(path / "run.json"), "metrics": metrics}
    candidates, airports = {}, []
    for variant in config["variants"]:
        folds, ratios = {}, []
        for fold in config["development_folds"]:
            path, run = experiments[(variant, fold)]
            predictions = pd.read_parquet(path / "predictions.parquet")
            if identity_hash(predictions[ID]) != original["folds"][fold]["identity_sha256"]["score"]:
                raise ValueError("Candidate prediction population changed.")
            metric = evaluate(truths[fold], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
            if not np.isclose(metric["overall"]["rmse_seconds"], read_json(path / "metrics.json")["overall"]["rmse_seconds"], rtol=1e-12):
                raise ValueError("Saved candidate RMSE does not reproduce.")
            reference_rmse = reference_reports[fold]["metrics"]["overall"]["rmse_seconds"]
            ratio = metric["overall"]["rmse_seconds"]/reference_rmse
            ratios.append(ratio)
            folds[fold] = {"run_id": run["run_id"], "run_sha256": sha256_file(path / "run.json"),
                           "component_iterations": run["component_iterations"], "metrics": metric,
                           "rmse_ratio_to_phase3": ratio, "rmse_improvement_seconds": reference_rmse-metric["overall"]["rmse_seconds"],
                           "paired_day_uncertainty": paired_days(truths[fold], predictions, reference_predictions[fold], config["seed"]),
                           "resources": run["resources"]}
            for airport, details in metric["by_airport"].items():
                airports.append({"variant": variant, "fold": fold, "airport": airport, **details})
        candidates[variant] = {"recipe": config["variants"][variant], "folds": folds,
                               "improves_both_folds": all(r < 1 for r in ratios), "mean_relative_rmse": float(np.mean(ratios))}
    eligible = [v for v, c in candidates.items() if c["improves_both_folds"]]
    chosen = min(eligible, key=lambda v: candidates[v]["mean_relative_rmse"]) if eligible else None
    pipeline, iterations = None, {}
    if chosen:
        pipelines = [read_json(experiments[(chosen, fold)][0] / "pipeline.json") for fold in config["development_folds"]]
        if (pipelines[0]["patches"] != pipelines[1]["patches"] or pipelines[0]["default"] != pipelines[1]["default"]
                or set(pipelines[0]["nodes"]) != set(pipelines[1]["nodes"])):
            raise ValueError("Selected fold pipeline structures differ.")
        pipeline = deepcopy(pipelines[0])
        for name, node in pipeline["nodes"].items():
            other = pipelines[1]["nodes"][name]
            if any(node[k] != other[k] for k in ["schema", "route", "offset", "settings"]):
                raise ValueError("Selected fold component settings differ.")
            iterations[name] = int(np.rint(np.median([node["iterations"], other["iterations"]])))
    selection = {
        "created_at_utc": utc_now(), "selected_model": "lightgbm", "selected_variant": chosen,
        "refit_pipeline": pipeline, "refit_component_iterations": iterations,
        "refit_months": config["refit_months"], "audit_used": False, "rule": config["selection"],
        "configuration": config, "config_sha256": config_hash, "config_path": args.config.as_posix(),
        "model_code_sha256": code["sha256"], "selection_code_sha256": sha256_file(Path(__file__)),
        "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
        "references": reference_reports, "candidates": candidates,
    }
    write_json(root / args.output, selection)
    report_dir = root / args.report_dir
    write_json(report_dir / "development_results.json", selection)
    write_json(report_dir / "feature_manifest.json", manifest)
    pd.DataFrame(airports).to_csv(report_dir / "airport_errors.csv", index=False)
    print("Selected:", chosen, "component iterations:", iterations)
    for variant, details in candidates.items():
        print(variant, {f: round(d["metrics"]["overall"]["rmse_seconds"], 6) for f, d in details["folds"].items()},
              "improves_both:", details["improves_both_folds"])


if __name__ == "__main__":
    main()
