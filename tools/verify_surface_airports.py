"""Verify all airport populations, native models and complete candidate metrics."""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID
from prc2026.phase1 import validate_inputs
from prc2026.validation import identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_nmclock.features import route_mask
from prc2026_timeline.bundle import load_bundle
from prc2026_timeline.features import FEATURE_NAMES, CATEGORICAL_COLUMNS
from prc2026_timeline.encoding import contract
from prc2026_timeline.inference import node_prediction
from prc2026_surface_airports.training import checked_runs, model_code_receipt
from prc2026_surface_airports.inference import airport_prediction
from select_surface_airports import collect
from verify_phase5 import same_metrics


def compare_saved_metrics(*, rebuilt, saved):
    """The saved JSON is the canonical string-keyed expected representation."""
    same_metrics(rebuilt, saved)


def verify(base, runs, timeline_runs, feature_report):
    config = read_json(ROOT / "configs/surface_airports_v1.json")
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    if validate_inputs(ROOT) != manifest["descriptor"]["input_files"]:
        raise ValueError("Original supplied inputs changed.")
    prior_features = read_json(feature_report)
    if not prior_features["passed"] or prior_features["bundle_id"] != manifest["bundle_id"] or prior_features["manifest_sha256"] != sha256_file(base / "manifest.json"):
        raise ValueError("The unchanged timeline features must already be independently verified.")
    saved = read_json(runs / "selection.json")
    rebuilt, predictions = collect(ROOT, base, ROOT / "configs/surface_airports_v1.json", runs, timeline_runs)
    for key in ["config_sha256", "selection_code_sha256", "paired_day_code_sha256", "model_code_sha256", "bundle_id", "native_fit_count", "candidate_count", "timeline_selection_sha256", "selected_candidate", "refit_iterations"]:
        if saved[key] != rebuilt[key]:
            raise ValueError("Airport selection no longer reproduces: " + key)
    for key in ["candidates", "reference_metrics", "selected_paired_days"]:
        compare_saved_metrics(rebuilt=rebuilt[key], saved=saved[key])
    experiments = checked_runs(ROOT, runs, config, manifest)
    native = {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent / "splits" / (fold + ".parquet"))
        scoring, _ = load_partition(base, parent, catalog, assignment, "score", FEATURE_NAMES)
        fitting, _ = load_partition(base, parent, catalog, assignment, "fit", list(dict.fromkeys(["airport", "nmclock_available", "nmclock_proxy_s", *CATEGORICAL_COLUMNS])))
        stopping, _ = load_partition(base, parent, catalog, assignment, "early_stop", ["airport", "nmclock_available", "nmclock_proxy_s"])
        fit, stop, score = [frame.loc[route_mask(frame)].reset_index(drop=True) for frame in [fitting, stopping, scoring]]
        run, _ = experiments[fold]
        nodes = read_json(run / "nodes.json")
        expected = pd.read_parquet(run / "predictions.parquet").set_index(ID)["prediction"]
        native[fold], samples = {}, []
        for airport in config["airports"]:
            node = nodes[airport]
            a, b, c = [frame.loc[frame.airport.eq(airport)].reset_index(drop=True) for frame in [fit, stop, score]]
            if node["schema"] != "timeline" or node["spec"] != config["native_spec"] or node["settings"] != config["settings"] or node["target_filter"] != "none":
                raise ValueError("An airport model's frozen native contract changed.")
            if node["fit_rows"] != len(a) or node["early_stop_rows"] != len(b) or node["fit_ids_sha256"] != identity_hash(a[ID]) or node["early_stop_ids_sha256"] != identity_hash(b[ID]):
                raise ValueError("An airport component used different fitting or stopping identities.")
            if not 1 <= node["iterations"] <= config["settings"]["lightgbm"]["rounds"]:
                raise ValueError("Invalid October-selected airport iteration count.")
            encoder = read_json(run / node["encoder_path"])
            names, cats = contract(config["native_spec"])
            vocab = {name: sorted(set(a[name].astype("string").fillna("__MISSING__")) | {"__MISSING__"}) for name in cats}
            if encoder["feature_names"] != names or encoder["categorical_columns"] != cats or encoder["vocabularies"] != vocab:
                raise ValueError("An airport encoder is not fitting-only.")
            special = c.nmclock_proxy_s.lt(0) | c.nmclock_proxy_s.gt(10800) | c.nmclock_clock_difference_s.abs().gt(3600)
            indexes = np.unique(np.r_[np.linspace(0, len(c) - 1, min(1000, len(c)), dtype=int), np.flatnonzero(special)])
            sample = c.iloc[indexes].reset_index(drop=True)
            actual = node_prediction(run, node, sample, config["threads"])
            delta = float(np.max(np.abs(actual - expected.loc[sample[ID]].to_numpy())))
            if not np.isfinite(delta) or delta > 1e-8:
                raise ValueError("An airport native model fails prediction reconstruction.")
            native[fold][airport] = {"rows": len(sample), "maximum_difference_seconds": delta, "fit_encoder_verified": True,
                                     "fitting_rows": len(a), "early_stop_rows": len(b), "iterations": node["iterations"]}
            samples.append(sample)
        all_samples = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=19).reset_index(drop=True)
        delta = float(np.max(np.abs(airport_prediction(run, nodes, all_samples, config["threads"]) - expected.loc[all_samples[ID]].to_numpy())))
        if delta > 1e-8:
            raise ValueError("Combined airport routing changes predictions.")
        missing = ~route_mask(scoring)
        reference = scoring[[ID]].merge(pd.read_parquet(runs / "references" / fold / "v6.parquet"), on=ID, validate="one_to_one")["prediction"]
        for (name, prediction_fold), prediction in predictions.items():
            if prediction_fold == fold and not np.array_equal(prediction.loc[missing, "prediction"], reference.loc[missing]):
                raise ValueError("A complete candidate changes the missing-clock v6 fallback.")
    return {"created_at_utc": utc_now(), "passed": True, "audit_used": False, "bundle_id": manifest["bundle_id"],
            "model_code_sha256": model_code_receipt(ROOT)["sha256"], "selection_sha256": sha256_file(runs / "selection.json"),
            "verifier_code_sha256": sha256_file(Path(__file__)), "native_fit_count": 20, "candidate_count": 4,
            "selected_candidate": saved["selected_candidate"], "refit_iterations": saved["refit_iterations"],
            "native_prediction_checks": native, "feature_verification_sha256": sha256_file(feature_report),
            "complete_rmse_seconds": {name: {fold: row["metrics"]["overall"]["rmse_seconds"] for fold, row in item["folds"].items()} for name, item in saved["candidates"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/surface_airports"))
    parser.add_argument("--timeline-runs", type=Path, default=Path("runs/timeline"))
    parser.add_argument("--feature-report", type=Path, default=Path("docs/timeline/feature_verification.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/surface_airports/verification.json"))
    args = parser.parse_args()
    destination = ROOT / args.output
    if destination.exists():
        raise ValueError("Preserve the original airport verification report.")
    report = verify(ROOT / "data/timeline/ce42240e5e43eee1713f", ROOT / args.runs, ROOT / args.timeline_runs, ROOT / args.feature_report)
    write_json(destination, report)
    print({"passed": report["passed"], "native_fit_count": 20, "selected_candidate": report["selected_candidate"]})


if __name__ == "__main__":
    main()
