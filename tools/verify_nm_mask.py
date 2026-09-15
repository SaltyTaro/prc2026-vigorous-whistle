"""Check full populations, fitting-only encoders, native reloads and all blends."""
from copy import deepcopy
from pathlib import Path
import argparse
import gc
import math
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, write_json, sha256_file, utc_now, verify_receipts
from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash
from prc2026_weather.encoding import fit_encoder
from prc2026_nm_mask.data import load_job
from prc2026_nm_mask.common import configurations, sample_weights, code_receipt
from prc2026_nm_mask.native import load_native, predict_native
from select_nm_mask import collect


def nested_equal(expected, actual, location="receipt"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"Recomputed receipt type differs at {location}.")
        # JSON serializes integer month keys as strings. Normalize that
        # representation, rejecting collisions rather than discarding a value.
        original_count = len(actual)
        actual = {str(key): value for key, value in actual.items()}
        if len(actual) != original_count or set(expected) != set(actual):
            raise ValueError(f"Recomputed receipt keys differ at {location}.")
        for key in expected:
            nested_equal(expected[key], actual[key], location + "/" + str(key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Recomputed receipt lengths differ at {location}.")
        for index, (left, right) in enumerate(zip(expected, actual)):
            nested_equal(left, right, location + "/" + str(index))
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-11, abs_tol=1e-9):
            raise ValueError(f"Recomputed numeric receipt differs at {location}: {expected}, {actual}")
    elif expected != actual:
        raise ValueError(f"A recomputed receipt value differs at {location}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/azure_parallel_nm_mask"))
    parser.add_argument("--selection", type=Path, default=Path("docs/nm_mask/development_results.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/nm_mask/local_verification.json"))
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise ValueError("Use a fresh verification report.")
    runs = ROOT / args.runs
    selection = read_json(ROOT / args.selection)
    _, config = configurations(ROOT)
    rebuilt, candidates = collect(ROOT, runs)
    for key in ["config_sha256", "model_code_sha256", "job_index_sha256", "native_fit_count", "candidate_count",
                "development_runs", "reference_metrics", "candidates", "selected_candidate", "refit_iterations"]:
        nested_equal(selection[key], rebuilt[key])
    pointer = read_json(ROOT / "data/nm_mask/latest.json")
    base = ROOT / pointer["bundle"]
    manifest = read_json(base / "manifest.json")
    route_hashes = {r["path"]: r["sha256"] for r in manifest["artifacts"] if r["path"].startswith("routing/")}
    jobs = read_json(ROOT / "docs/nm_mask/jobs_bound.json")["jobs"]
    predictions, metadata, native_reports = {}, {}, {}
    for item in jobs:
        job, frames = load_job(ROOT, ROOT / item["path"])
        if job["feature_manifest_sha256"] != sha256_file(base / "manifest.json"):
            raise ValueError("The source feature manifest differs from the fitted job.")
        for role in ["fit", "early_stop", "score"]:
            if job["original_partition_identity"][role]["original_ids_sha256"] != manifest["frozen_folds"][item["fold"]]["identity_sha256"][role]:
                raise ValueError("The original protected split identity changed.")
        directory = runs / item["job_id"]
        run = read_json(directory / "run.json")
        verify_receipts(directory, run["artifacts"])
        variant = config["variants"][item["variant"]]
        model, encoder, node = load_native(directory)
        expected_encoder = fit_encoder(frames["fit_features"], {"kind": variant["kind"], "features": "ground_weather"})
        expected_encoder["fitted_on"] = "fitting_partition_only"
        if encoder != expected_encoder:
            raise ValueError("The saved categorical vocabulary is not fitting-only.")
        expected_params = deepcopy(variant["settings"][variant["kind"]]["parameters"])
        if variant["kind"] == "lightgbm":
            expected_params.update(seed=config["seed"], num_threads=16, bagging_seed=config["seed"],
                                   feature_fraction_seed=config["seed"], data_random_seed=config["seed"])
            curve = read_json(directory / "history.json")["early_stop"]["rmse"]
        else:
            expected_params.update(iterations=variant["settings"]["catboost"]["rounds"], random_seed=config["seed"], thread_count=16)
            curve = read_json(directory / "history.json")["validation"]["RMSE"]
        if node["native_parameters"] != expected_params or node["iterations"] != int(np.argmin(curve)) + 1:
            raise ValueError("Native settings or the October-selected count changed.")
        weighted_ids = pd.read_parquet(directory / "fit_ids_weights.parquet")
        np.testing.assert_array_equal(weighted_ids[ID], frames["fit_features"][ID])
        np.testing.assert_array_equal(weighted_ids.natural_missing, frames["fit_truth"].natural_missing)
        weights = sample_weights(frames["fit_truth"].natural_missing.to_numpy(dtype=bool), variant["fit_population"])
        np.testing.assert_array_equal(weighted_ids.sample_weight, weights)
        np.testing.assert_array_equal(pd.read_parquet(directory / "early_stop_ids.parquet")[ID], frames["early_stop_features"][ID])
        predicted = predict_native(model, encoder, frames["score_features"], node["iterations"])
        stored = pd.read_parquet(directory / "component_predictions.parquet")
        np.testing.assert_array_equal(stored[ID], frames["score_features"][ID])
        difference = float(np.max(np.abs(predicted - stored.prediction.to_numpy())))
        if difference > 1e-8:
            raise ValueError("Native reconstruction changed an eligible prediction.")
        predictions[(item["fold"], item["variant"])] = predicted
        native_reports[item["job_id"]] = {"worker": item["worker"], "rows": len(predicted), "maximum_difference_seconds": difference,
            "fit_rows": len(weights), "fit_ids_sha256": identity_hash(weighted_ids[ID]),
            "fitting_only_encoder_verified": True, "sample_weights_verified": True,
            "stopping_ids_verified": True, "iterations": node["iterations"]}
        if item["fold"] not in metadata:
            truth = frames["score_truth"]
            pieces = []
            for source in sorted(truth.source_file.unique()):
                path = "routing/" + source
                if sha256_file(base / path) != route_hashes[path]:
                    raise ValueError("Original routing projection changed.")
                routing = pd.read_parquet(base / path)
                pieces.append(routing.loc[routing[ID].isin(truth[ID])])
            aligned = truth[[ID]].merge(pd.concat(pieces, ignore_index=True), on=ID, how="left", validate="one_to_one", sort=False)
            if aligned[["nm_missing", "nmclock_available"]].isna().any().any():
                raise ValueError("Incomplete original routing projection.")
            eligible = (aligned.nm_missing.to_numpy() == 1) & (aligned.nmclock_available.to_numpy() == 0)
            np.testing.assert_array_equal(eligible, truth.natural_missing)
            metadata[item["fold"]] = (truth, frames["reference"], eligible)
        del frames, model, encoder, expected_encoder, weighted_ids
        gc.collect()
    independent_maximum, complete_checks = 0., 0
    for name, candidate in selection["candidates"].items():
        family = config["candidate_families"][candidate["family"]]
        weight = candidate["candidate_weight"]
        for fold in config["development_folds"]:
            path = f"candidate_predictions/{fold}_{name}.parquet"
            if sha256_file(runs / path) != selection["candidate_prediction_sha256"][path]:
                raise ValueError("A complete candidate artifact changed.")
            saved = pd.read_parquet(runs / path)
            pd.testing.assert_frame_equal(saved, candidates[(fold, name)], check_exact=True)
            truth, reference, eligible = metadata[fold]
            family_values = sum(share * predictions[(fold, variant)] for variant, share in family.items())
            independent = reference.prediction.to_numpy().copy()
            independent[eligible] = ((1. - weight) * independent[eligible] + weight * family_values)
            np.testing.assert_array_equal(saved.prediction.to_numpy()[~eligible], reference.prediction.to_numpy()[~eligible])
            difference = float(np.max(np.abs(independent - saved.prediction.to_numpy())))
            independent_maximum = max(independent_maximum, difference)
            if difference > 1e-8:
                raise ValueError("The independently reconstructed family weights differ.")
            error = truth[TARGET].to_numpy(dtype=float) - independent
            rmse = float(np.sqrt(np.dot(error, error) / len(error)))
            if abs(rmse - candidate["folds"][fold]["metrics"]["overall"]["rmse_seconds"]) > 1e-8:
                raise ValueError("A complete independent RMSE differs.")
            complete_checks += 1
    report = {"created_at_utc": utc_now(), "passed": True, "native_fit_count": 8,
        "native_prediction_rows": sum(v["rows"] for v in native_reports.values()), "native_predictions": native_reports,
        "complete_candidate_fold_metrics_reproduced": complete_checks,
        "independent_blend_maximum_difference_seconds": independent_maximum,
        "selected_candidate": selection["selected_candidate"], "refit_iterations": selection["refit_iterations"],
        "model_code_sha256": code_receipt(ROOT)["sha256"], "selection_sha256": sha256_file(ROOT / args.selection),
        "verifier_code_sha256": sha256_file(Path(__file__)), "audit_used": False}
    write_json(output, report)
    print({k: v for k, v in report.items() if k != "native_predictions"}, flush=True)


if __name__ == "__main__":
    main()
