"""Reproduce all twelve fixed candidates after all eight jobs finish."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, write_json, sha256_file, verify_receipts, utc_now
from prc2026.features import ID
from prc2026.validation import evaluate, identity_hash
from prc2026_nm_mask.common import configurations, compose, code_receipt


def collect(root, runs):
    _, config = configurations(root)
    jobs = read_json(root / "docs/nm_mask/jobs_bound.json")["jobs"]
    reports, components, data, references = {}, {}, {}, {}
    model_hash = code_receipt(root)["sha256"]
    for item in jobs:
        job = read_json(root / item["path"])
        if sha256_file(root / item["path"]) != item["sha256"]:
            raise ValueError("The bound job definition changed.")
        directory = runs / item["job_id"]
        report = read_json(directory / "run.json")
        if (report["job_sha256"] != item["sha256"] or report["model_code"]["sha256"] != model_hash
                or report["audit_used"] or report["new_native_fits"] != 1
                or report["worker"] != item["worker"] or report["fold"] != item["fold"] or report["variant"] != item["variant"]):
            raise ValueError("A completed run does not match its declared worker and source.")
        verify_receipts(directory, report["artifacts"])
        fold = item["fold"]
        if fold not in data:
            for key in ["score_truth", "reference", "reference_metrics"]:
                path = job["paths"][key]
                if sha256_file(root / path) != job["files"][path]:
                    raise ValueError("A complete scoring reference changed.")
            truth = pd.read_parquet(root / job["paths"]["score_truth"])
            reference = pd.read_parquet(root / job["paths"]["reference"])
            np.testing.assert_array_equal(truth[ID], reference[ID])
            if identity_hash(truth[ID]) != job["original_partition_identity"]["score"]["original_ids_sha256"]:
                raise ValueError("Complete scoring IDs changed.")
            data[fold] = (truth, reference, job["ranking_month_weights"])
            references[fold] = evaluate(truth, reference, job["ranking_month_weights"])
            expected = read_json(root / job["paths"]["reference_metrics"])
            if abs(references[fold]["overall"]["rmse_seconds"] - expected["overall"]["rmse_seconds"]) > 1e-9:
                raise ValueError("V7 baseline no longer reproduces.")
        truth, reference, month_weights = data[fold]
        eligible = truth.natural_missing.to_numpy(dtype=bool)
        component = pd.read_parquet(directory / "component_predictions.parquet")
        np.testing.assert_array_equal(truth.loc[eligible, ID], component[ID])
        full = pd.DataFrame({ID: truth[ID], "prediction": compose(reference.prediction, component.prediction, eligible, 1.)})
        pd.testing.assert_frame_equal(full, pd.read_parquet(directory / "predictions.parquet"), check_exact=True)
        metrics = evaluate(truth, full, month_weights)
        if abs(metrics["overall"]["rmse_seconds"] - report["complete_rmse_seconds"]) > 1e-9:
            raise ValueError("A complete native score no longer reproduces.")
        components[(fold, item["variant"])] = component.prediction.to_numpy()
        reports[item["job_id"]] = {"worker": item["worker"], "fold": fold, "variant": item["variant"],
            "run_sha256": sha256_file(directory / "run.json"), "iterations": report["iterations"],
            "complete_rmse_seconds": metrics["overall"]["rmse_seconds"], "resources": report["resources"]}
    expected = {(fold, variant) for fold in config["development_folds"] for variant in config["variants"]}
    if set(components) != expected or len(reports) != 8:
        raise ValueError("Every declared job must finish before selection.")
    candidates, predictions = {}, {}
    for family, weights in config["candidate_families"].items():
        if set(weights) - set(config["variants"]) or abs(sum(weights.values()) - 1.) > 1e-12:
            raise ValueError("Invalid declared family weights.")
        for weight in config["candidate_weights"]:
            name = f"{family}_v7_{int(weight * 100)}"
            folds, ratios = {}, []
            for fold in config["development_folds"]:
                truth, reference, month_weights = data[fold]
                eligible = truth.natural_missing.to_numpy(dtype=bool)
                values = np.zeros(int(eligible.sum()), dtype=float)
                for variant, share in weights.items():
                    values += share * components[(fold, variant)]
                full = compose(reference.prediction, values, eligible, weight)
                np.testing.assert_array_equal(full[~eligible], reference.prediction.to_numpy()[~eligible])
                frame = pd.DataFrame({ID: truth[ID], "prediction": full})
                metrics = evaluate(truth, frame, month_weights)
                ratio = metrics["overall"]["rmse_seconds"] / references[fold]["overall"]["rmse_seconds"]
                folds[fold] = {"metrics": metrics, "rmse_ratio_to_v7": ratio,
                               "eligible_rows": int(eligible.sum()), "unchanged_rows": int((~eligible).sum())}
                ratios.append(ratio)
                predictions[(fold, name)] = frame
            candidates[name] = {"family": family, "candidate_weight": weight, "folds": folds,
                "mean_relative_rmse": float(np.mean(ratios)),
                "eligible": all(value <= 1. - config["minimum_improvement_fraction_each_fold"] for value in ratios)}
    eligible = [name for name, candidate in candidates.items() if candidate["eligible"]]
    selected = min(eligible, key=lambda name: (candidates[name]["mean_relative_rmse"], name)) if eligible else None
    counts = None
    if selected:
        variants = config["candidate_families"][candidates[selected]["family"]]
        counts = {variant: int(np.rint(np.median([r["iterations"] for r in reports.values() if r["variant"] == variant])))
                  for variant in variants}
    report = {"created_at_utc": utc_now(), "configuration": config, "config_sha256": sha256_file(root / "configs/nm_mask_v1.json"),
        "model_code_sha256": model_hash, "job_index_sha256": sha256_file(root / "docs/nm_mask/jobs_bound.json"),
        "native_fit_count": 8, "candidate_count": 12, "development_runs": reports,
        "reference_metrics": references, "candidates": candidates, "selected_candidate": selected,
        "refit_iterations": counts, "audit_used": False, "target_filter": "none", "prediction_postprocess": "none",
        "submission_uploaded": False, "selection_code_sha256": sha256_file(Path(__file__))}
    return report, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/azure_parallel_nm_mask"))
    parser.add_argument("--output", type=Path, default=Path("docs/nm_mask/development_results.json"))
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise ValueError("A selection receipt already exists; preserve it.")
    report, predictions = collect(ROOT, ROOT / args.runs)
    destination = ROOT / args.runs / "candidate_predictions"
    destination.mkdir(exist_ok=False)
    hashes = {}
    for (fold, name), frame in predictions.items():
        path = destination / f"{fold}_{name}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        hashes[path.relative_to(ROOT / args.runs).as_posix()] = sha256_file(path)
    report["candidate_prediction_sha256"] = hashes
    write_json(output, report)
    print({"selected_candidate": report["selected_candidate"], "complete_scores": {
        name: {fold: item["metrics"]["overall"]["rmse_seconds"] for fold, item in candidate["folds"].items()}
        for name, candidate in report["candidates"].items()}}, flush=True)


if __name__ == "__main__":
    main()
