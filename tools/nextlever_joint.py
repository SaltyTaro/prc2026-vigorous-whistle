"""Evaluate two predeclared combinations of complementary new feature routes."""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.phase1 import validate_inputs
from prc2026.validation import evaluate, identity_hash
from prc2026_nmclock.bundle import load_bundle
from prc2026_nmclock.features import route_mask
from prc2026_phase3.bundle import load_partition
from prc2026_ladder.inference import compose
from prc2026_ladder.training import aligned_predictions
from prc2026_timeline.training import model_code_receipt as timeline_code
from prc2026_arrivalctx.training import model_code_receipt as arrival_code
from select_phase4 import paired_days
from verify_phase5 import same_metrics


def combine_routes(reference, timeline, arrival, available, weight):
    reference, timeline, arrival = [np.asarray(values, dtype=float) for values in [reference, timeline, arrival]]
    available = np.asarray(available)
    if reference.ndim != 1 or timeline.shape != reference.shape or arrival.shape != reference.shape or available.shape != reference.shape or available.dtype != bool:
        raise ValueError("Route combinations need aligned vectors and a boolean availability mask.")
    if not all(np.isfinite(values).all() for values in [reference, timeline, arrival]) or weight not in [.5, 1.]:
        raise ValueError("Only the two predeclared finite convex combinations are allowed.")
    if not np.array_equal(timeline[~available], reference[~available]) or not np.array_equal(arrival[available], reference[available]):
        raise ValueError("The parent experiments no longer preserve complementary v6 routes.")
    replacement = np.where(available, timeline, arrival)
    return replacement if weight == 1 else (1 - weight) * reference + weight * replacement


def checked_source(runs, verification_path, count, code):
    selection = read_json(runs / "selection.json")
    verified = read_json(verification_path)
    if (selection["audit_used"] or not verified["passed"] or verified["audit_used"]
            or selection["candidate_count"] != count or verified["candidate_count"] != count
            or verified["selection_sha256"] != sha256_file(runs / "selection.json")
            or selection["model_code_sha256"] != code or verified["model_code_sha256"] != code):
        raise ValueError("A complete verified parent campaign is required.")
    for name, folds in selection["development_runs"].items():
        for fold, saved in folds.items():
            run = runs / saved["run_id"]
            if sha256_file(run / "run.json") != saved["run_sha256"] or verified["run_sha256"][fold][name] != saved["run_sha256"]:
                raise ValueError("A producing parent run changed.")
            verify_receipts(run, read_json(run / "run.json")["artifacts"])
    return selection


def recipe_prediction(runs, selection, name, fold, features, reference):
    recipe = selection["candidates"][name]
    names = {key for weights in recipe["routes"].values() for key in weights} - {"v6"}
    components = {}
    for component in names:
        if component in {"v6_global", "v6_airports"}:
            path = runs / "references" / fold / (component + ".parquet")
        else:
            path = runs / selection["development_runs"][component][fold]["run_id"] / "predictions.parquet"
        components[component] = aligned_predictions(features, pd.read_parquet(path))["prediction"].to_numpy()
    return compose(reference, components, route_mask(features), recipe["routes"], recipe["new_weight"])


def collect(timeline_runs, arrival_runs, timeline_verification, arrival_verification):
    config_path = ROOT / "configs/nextlever_joint_v1.json"
    config = read_json(config_path)
    if (config["audit_used"] or config["development_folds"] != ["seasonal", "forward"]
            or config["blend_weights_new_candidate"] != [.5, 1.] or config["minimum_improvement_fraction_each_fold"] != .005
            or config["timeline_recipe"] != "combined_mean_v6_100" or config["arrival_recipe"] != "arrival_mean_v6_100"
            or config["new_native_fits"] != 0):
        raise ValueError("The separate two-recipe declaration changed.")
    for path, expected in [(ROOT / config["reference_selection"], config["reference_selection_sha256"]),
                           (ROOT / "configs/timeline_v1.json", config["timeline_configuration_sha256"]),
                           (ROOT / "configs/arrivalctx_v1.json", config["arrival_configuration_sha256"])]:
        if sha256_file(path) != expected:
            raise ValueError("An original declaration or v6 selection changed.")
    timeline = checked_source(timeline_runs, timeline_verification, 10, timeline_code(ROOT)["sha256"])
    arrival = checked_source(arrival_runs, arrival_verification, 6, arrival_code(ROOT)["sha256"])
    if sum(len(folds) for folds in timeline["development_runs"].values()) != 8 or sum(len(folds) for folds in arrival["development_runs"].values()) != 4:
        raise ValueError("All twelve preceding native fits are required.")
    expected_timeline = {"clock_available": {"lgb_combined": .25, "cat_combined": .25, "v6_airports": .5}, "clock_missing": {"v6": 1.}}
    expected_arrival = {"clock_available": {"v6": 1.}, "clock_missing": {"lgb_arrival": .5, "cat_arrival": .5}}
    for selection, name, expected in [(timeline, config["timeline_recipe"], expected_timeline), (arrival, config["arrival_recipe"], expected_arrival)]:
        if selection["candidates"][name]["routes"] != expected or selection["candidates"][name]["new_weight"] != 1.:
            raise ValueError("A fixed parent family changed.")
    base = ROOT / "data/nmclock/72e86725e23367ede5b5"
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    if manifest["bundle_id"] != config["feature_bundle_id"] or validate_inputs(ROOT) != manifest["descriptor"]["input_files"]:
        raise ValueError("Original supplied data or clock routing changed.")
    truths, references, features, replacements, ref_metrics, source_errors = {}, {}, {}, {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent / "splits" / (fold + ".parquet"))
        features[fold], truths[fold] = load_partition(base, parent, catalog, assignment, "score", ["nmclock_available", "nmclock_proxy_s"])
        if identity_hash(features[fold][ID]) != original["folds"][fold]["identity_sha256"]["score"]:
            raise ValueError("Protected scoring identities changed.")
        reference = aligned_predictions(features[fold], pd.read_parquet(timeline_runs / "references" / fold / "v6.parquet"))
        other_reference = aligned_predictions(features[fold], pd.read_parquet(arrival_runs / "references" / fold / "v6.parquet"))
        if not np.array_equal(reference["prediction"], other_reference["prediction"]):
            raise ValueError("The complete parent v6 reference predictions differ.")
        references[fold] = reference
        seasonal_weights = original["ranking_month_weights"] if fold == "seasonal" else None
        ref_metrics[fold] = evaluate(truths[fold], reference, seasonal_weights)
        same_metrics(ref_metrics[fold], timeline["reference_metrics"][fold])
        same_metrics(ref_metrics[fold], arrival["reference_metrics"][fold])
        t = recipe_prediction(timeline_runs, timeline, config["timeline_recipe"], fold, features[fold], reference["prediction"])
        a = recipe_prediction(arrival_runs, arrival, config["arrival_recipe"], fold, features[fold], reference["prediction"])
        for values, selection, name in [(t, timeline, config["timeline_recipe"]), (a, arrival, config["arrival_recipe"])]:
            recomputed = evaluate(truths[fold], pd.DataFrame({ID: features[fold][ID], "prediction": values}), seasonal_weights)
            same_metrics(recomputed, selection["candidates"][name]["folds"][fold]["metrics"])
        replacements[fold] = (t, a)
        source_errors[fold] = {"timeline_rmse_seconds": timeline["candidates"][config["timeline_recipe"]]["folds"][fold]["metrics"]["overall"]["rmse_seconds"],
                               "arrival_rmse_seconds": arrival["candidates"][config["arrival_recipe"]]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]}
    candidates, predictions = {}, {}
    for weight in config["blend_weights_new_candidate"]:
        name, folds, ratios = "joint_mean_v6_" + str(int(weight * 100)), {}, []
        for fold in config["development_folds"]:
            t, a = replacements[fold]
            values = combine_routes(references[fold]["prediction"], t, a, route_mask(features[fold]), weight)
            prediction = pd.DataFrame({ID: features[fold][ID], "prediction": values})
            metrics = evaluate(truths[fold], prediction, original["ranking_month_weights"] if fold == "seasonal" else None)
            ratio = metrics["overall"]["rmse_seconds"] / ref_metrics[fold]["overall"]["rmse_seconds"]
            folds[fold] = {"metrics": metrics, "rmse_ratio_to_v6": ratio}
            predictions[(name, fold)] = prediction
            ratios.append(ratio)
        candidates[name] = {"new_weight": weight, "folds": folds, "mean_relative_rmse": float(np.mean(ratios)), "eligible": all(ratio <= .995 for ratio in ratios)}
    eligible = [name for name, row in candidates.items() if row["eligible"]]
    selected = min(eligible, key=lambda name: (candidates[name]["mean_relative_rmse"], name)) if eligible else None
    paired, iterations = {}, {}
    if selected:
        for fold in config["development_folds"]:
            paired[fold] = paired_days(truths[fold], predictions[(selected, fold)], references[fold], 20260913)
        for prefix, source, names in [("timeline", timeline, ["lgb_combined", "cat_combined"]), ("arrivalctx", arrival, ["lgb_arrival", "cat_arrival"])]:
            for name in names:
                iterations[prefix + "/" + name] = int(np.rint(np.median([source["development_runs"][name][fold]["iterations"] for fold in config["development_folds"]])))
    report = {"created_at_utc": utc_now(), "configuration": config, "config_sha256": object_hash(config),
              "code_sha256": sha256_file(Path(__file__)), "paired_day_code_sha256": sha256_file(ROOT / "tools/select_phase4.py"),
              "timeline_selection_sha256": sha256_file(timeline_runs / "selection.json"), "arrival_selection_sha256": sha256_file(arrival_runs / "selection.json"),
              "source_model_code_sha256": {"timeline": timeline_code(ROOT)["sha256"], "arrivalctx": arrival_code(ROOT)["sha256"]},
              "bundle_id": manifest["bundle_id"], "audit_used": False, "new_native_fits": 0,
              "candidate_count": 2, "candidates": candidates, "selected_candidate": selected,
              "reference_metrics": ref_metrics, "source_recipe_rmse_seconds": source_errors, "selected_paired_days": paired,
              "refit_iterations": iterations, "refit_months": [f"2025-{month:02}" for month in range(1, 12)],
              "interpretation": "Two separately predeclared combinations of complementary feature routes. Complete reused development scores, not official results or the protected December audit. No additional model fit or weight search."}
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-runs", type=Path, default=Path("runs/timeline"))
    parser.add_argument("--arrival-runs", type=Path, default=Path("runs/arrivalctx"))
    parser.add_argument("--timeline-verification", type=Path, default=Path("docs/timeline/verification.json"))
    parser.add_argument("--arrival-verification", type=Path, default=Path("docs/arrivalctx/verification.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/nextlever_joint"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verification-output", type=Path, default=Path("docs/nextlever_joint/verification.json"))
    args = parser.parse_args()
    output = ROOT / args.output
    if not args.verify and ((output / "selection.json").exists() or (ROOT / "docs/nextlever_joint/development_results.json").exists()):
        raise ValueError("Preserve the previous joint selection.")
    report, predictions = collect(ROOT / args.timeline_runs, ROOT / args.arrival_runs, ROOT / args.timeline_verification, ROOT / args.arrival_verification)
    if args.verify:
        path = ROOT / args.verification_output
        if path.exists():
            raise ValueError("Preserve the previous joint verification.")
        saved = read_json(output / "selection.json")
        verify_receipts(output, saved["artifacts"])
        keys = ["config_sha256", "code_sha256", "paired_day_code_sha256", "timeline_selection_sha256", "arrival_selection_sha256", "source_model_code_sha256", "bundle_id", "candidate_count", "selected_candidate", "refit_iterations"]
        for key in keys:
            if report[key] != saved[key]:
                raise ValueError("Joint selection changed: " + key)
        for key in ["candidates", "reference_metrics", "source_recipe_rmse_seconds", "selected_paired_days"]:
            same_metrics(report[key], saved[key])
        for (name, fold), prediction in predictions.items():
            existing = pd.read_parquet(output / name / (fold + ".parquet"))
            pd.testing.assert_frame_equal(prediction, existing, check_exact=True)
        result = {"created_at_utc": utc_now(), "passed": True, "audit_used": False, "candidate_count": 2,
                  "selection_sha256": sha256_file(output / "selection.json"), "code_sha256": report["code_sha256"],
                  "selected_candidate": report["selected_candidate"], "maximum_prediction_difference_seconds": 0.,
                  "verified_prediction_rows": sum(len(value) for value in predictions.values()),
                  "complete_rmse_seconds": {name: {fold: row["metrics"]["overall"]["rmse_seconds"] for fold, row in item["folds"].items()} for name, item in report["candidates"].items()}}
        write_json(path, result)
        print(result)
    else:
        artifacts = []
        for (name, fold), prediction in predictions.items():
            path = output / name / (fold + ".parquet")
            if path.exists():
                raise ValueError("Preserve previous candidate predictions.")
            path.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(path, index=False, compression="zstd")
            artifacts.append({"path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        report["artifacts"] = artifacts
        write_json(output / "selection.json", report)
        write_json(ROOT / "docs/nextlever_joint/development_results.json", report)
        print({"selected": report["selected_candidate"], "candidates": {name: {fold: row["metrics"]["overall"]["rmse_seconds"] for fold, row in item["folds"].items()} for name, item in report["candidates"].items()}})


if __name__ == "__main__":
    main()
