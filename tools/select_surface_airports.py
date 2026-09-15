"""Select only after all twenty airport fits and four complete recipes finish."""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.validation import evaluate
from prc2026_phase3.bundle import load_partition
from prc2026_nmclock.features import route_mask
from prc2026_ladder.inference import compose
from prc2026_ladder.training import aligned_predictions, reference_selection, model_code_receipt as reference_code
from prc2026_timeline.bundle import load_bundle
from prc2026_surface_airports.training import checked_runs, model_code_receipt, validate_config
from select_phase4 import paired_days
from verify_phase5 import same_metrics


def collect(root, base, config_path, output, timeline_runs):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(root, config, original["descriptor"]["protocol"])
    if manifest["bundle_id"] != config["feature_bundle_id"] or sha256_file(base / "manifest.json") != config["feature_manifest_sha256"]:
        raise ValueError("Frozen feature inputs changed.")
    experiments = checked_runs(root, output, config, manifest)
    if set(experiments) != {"seasonal", "forward"}:
        raise ValueError("Both complete sets of ten airport fits must finish before selection.")
    prior, timeline = reference_selection(root, config), read_json(root / config["timeline_selection"])
    if sha256_file(timeline_runs / "selection.json") != config["timeline_selection_sha256"]:
        raise ValueError("The frozen timeline selection changed.")
    truths, features, references, components, ref_metrics, runs = {}, {}, {}, {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent / "splits" / (fold + ".parquet"))
        features[fold], truths[fold] = load_partition(base, parent, catalog, assignment, "score", ["nmclock_available", "nmclock_proxy_s"])
        ref_dir = output / "references" / fold
        receipt = read_json(ref_dir / "reference.json")
        if (receipt["source_run_sha256"] != prior["development_runs"][config["reference_variant"]][fold]["run_sha256"]
                or receipt["config_sha256"] != object_hash(config) or receipt["model_code_sha256"] != reference_code(root)["sha256"]):
            raise ValueError("An imported v6 reference changed.")
        verify_receipts(ref_dir, receipt["artifacts"])
        references[fold] = aligned_predictions(features[fold], pd.read_parquet(ref_dir / "v6.parquet"))
        ref_metrics[fold] = evaluate(truths[fold], references[fold], original["ranking_month_weights"] if fold == "seasonal" else None)
        same_metrics(ref_metrics[fold], timeline["reference_metrics"][fold])
        components[fold] = {"v6_global": aligned_predictions(features[fold], pd.read_parquet(ref_dir / "v6_global.parquet"))["prediction"].to_numpy()}
        path, report = experiments[fold]
        for role in ["fit", "early_stop", "score"]:
            if report[role + "_ids_sha256"] != original["folds"][fold]["identity_sha256"][role]:
                raise ValueError("Protected complete fold identities changed.")
        nodes = read_json(path / "nodes.json")
        if set(nodes) != set(config["airports"]) or report["iterations"] != {name: node["iterations"] for name, node in nodes.items()}:
            raise ValueError("The ten airport components are incomplete.")
        prediction = aligned_predictions(features[fold], pd.read_parquet(path / "predictions.parquet"))
        metrics = evaluate(truths[fold], prediction, original["ranking_month_weights"] if fold == "seasonal" else None)
        same_metrics(metrics, read_json(path / "metrics.json"))
        missing = ~route_mask(features[fold])
        if not np.array_equal(prediction.loc[missing, "prediction"], references[fold].loc[missing, "prediction"]):
            raise ValueError("The missing-clock fallback changed.")
        components[fold]["surface_airports"] = prediction["prediction"].to_numpy()
        runs[fold] = {"run_id": path.name, "run_sha256": sha256_file(path / "run.json"), "iterations": report["iterations"], "resources": report["resources"], "metrics": metrics}
        for name in ["lgb_combined", "cat_combined"]:
            source = timeline_runs / timeline["development_runs"][name][fold]["run_id"]
            if sha256_file(source / "run.json") != timeline["development_runs"][name][fold]["run_sha256"]:
                raise ValueError("An unchanged timeline global run changed.")
            verify_receipts(source, read_json(source / "run.json")["artifacts"])
            prediction = aligned_predictions(features[fold], pd.read_parquet(source / "predictions.parquet"))
            metrics = evaluate(truths[fold], prediction, original["ranking_month_weights"] if fold == "seasonal" else None)
            same_metrics(metrics, timeline["development_runs"][name][fold]["metrics"])
            components[fold][name] = prediction["prediction"].to_numpy()
    candidates, predictions = {}, {}
    for family_name, family in config["candidate_families"].items():
        for weight in config["blend_weights_new_candidate"]:
            name = family_name + "_v6_" + str(int(weight * 100))
            folds, ratios = {}, []
            for fold in config["development_folds"]:
                values = compose(references[fold]["prediction"], components[fold], route_mask(features[fold]), family, weight)
                prediction = pd.DataFrame({ID: features[fold][ID], "prediction": values})
                metrics = evaluate(truths[fold], prediction, original["ranking_month_weights"] if fold == "seasonal" else None)
                ratio = metrics["overall"]["rmse_seconds"] / ref_metrics[fold]["overall"]["rmse_seconds"]
                folds[fold] = {"metrics": metrics, "rmse_ratio_to_v6": ratio}
                ratios.append(ratio)
                predictions[(name, fold)] = prediction
            candidates[name] = {"family": family_name, "routes": family, "new_weight": weight, "folds": folds,
                                "mean_relative_rmse": float(np.mean(ratios)), "eligible": all(ratio <= .995 for ratio in ratios)}
    eligible = [name for name, row in candidates.items() if row["eligible"]]
    selected = min(eligible, key=lambda name: (candidates[name]["mean_relative_rmse"], name)) if eligible else None
    iterations, paired = {}, {}
    if selected:
        iterations = {"airports": {airport: int(np.rint(np.median([runs[fold]["iterations"][airport] for fold in config["development_folds"]]))) for airport in config["airports"]}, "globals": {}}
        if candidates[selected]["family"] == "full_timeline":
            iterations["globals"] = {name: int(np.rint(np.median([timeline["development_runs"][name][fold]["iterations"] for fold in config["development_folds"]]))) for name in ["lgb_combined", "cat_combined"]}
        for fold in config["development_folds"]:
            paired[fold] = paired_days(truths[fold], predictions[(selected, fold)], references[fold], config["seed"])
    report = {"created_at_utc": utc_now(), "configuration": config, "config_sha256": object_hash(config),
              "selection_code_sha256": sha256_file(Path(__file__)), "paired_day_code_sha256": sha256_file(root / "tools/select_phase4.py"),
              "model_code_sha256": model_code_receipt(root)["sha256"], "bundle_id": manifest["bundle_id"], "audit_used": False,
              "development_runs": runs, "native_fit_count": 20, "candidate_count": 4, "candidates": candidates,
              "reference_metrics": ref_metrics, "timeline_selection_sha256": config["timeline_selection_sha256"],
              "selected_candidate": selected, "refit_iterations": iterations, "selected_paired_days": paired,
              "refit_months": config["refit_months"], "rule": config["selection"]}
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/timeline/ce42240e5e43eee1713f"))
    parser.add_argument("--runs", type=Path, default=Path("runs/surface_airports"))
    parser.add_argument("--timeline-runs", type=Path, default=Path("runs/timeline"))
    args = parser.parse_args()
    destinations = [ROOT / args.runs / "selection.json", ROOT / "docs/surface_airports/development_results.json"]
    if any(path.exists() for path in destinations):
        raise ValueError("Preserve the original four-recipe selection.")
    report, _ = collect(ROOT, ROOT / args.bundle, ROOT / "configs/surface_airports_v1.json", ROOT / args.runs, ROOT / args.timeline_runs)
    for path in destinations:
        write_json(path, report)
    print({"selected": report["selected_candidate"], "candidates": {name: {fold: row["metrics"]["overall"]["rmse_seconds"] for fold, row in item["folds"].items()} for name, item in report["candidates"].items()}, "refit_iterations": report["refit_iterations"]})


if __name__ == "__main__":
    main()
