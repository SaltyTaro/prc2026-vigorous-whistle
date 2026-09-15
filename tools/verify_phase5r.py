"""Verify new bundle inheritance, cross-fitting, native predictions and selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_phase4.features import prediction_offset
from prc2026_phase5r.baseline import BASELINE_NAMES, aligned_truth, apply_map, fit_map, group_folds
from prc2026_phase5r.bundle import load_bundle
from prc2026_phase5r.features import CORE_NAMES, FEATURE_NAMES
from prc2026_phase5r.inference import node_prediction, pipeline_prediction, route_mask
from prc2026_phase5r.training import checked_runs, model_code_receipt, reference_predictions, validate_config
from select_phase5r import collect
from verify_phase5 import same_metrics


def maximum_difference(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Prediction/value comparison must have matching finite arrays.")
    return float(np.max(np.abs(left-right))) if left.size else 0.0


def sample_features(features: pd.DataFrame) -> pd.DataFrame:
    indexes = np.unique(np.r_[np.linspace(0, len(features)-1, min(1000, len(features)), dtype=int),
                               np.flatnonzero(features["lirf_missing_record"].eq(1).to_numpy())])
    return features.iloc[indexes].reset_index(drop=True)


def verify(root: Path, base: Path, config_path: Path, output: Path, prior: Path, followup: Path, require_refit: bool):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    code = model_code_receipt(root)
    feature_rows = 0
    for source, rows in manifest["rows"].items():
        features = pd.read_parquet(base/"features"/source)
        old = pd.read_parquet(root/manifest["previous_bundle"]/"features"/source, columns=[ID, *CORE_NAMES])
        if list(features) != [ID, *FEATURE_NAMES] or len(features) != rows or set(BASELINE_NAMES) & set(features):
            raise ValueError("New shared feature schema/row count changed.")
        pd.testing.assert_frame_equal(features[[ID, *CORE_NAMES]], old, check_exact=True)
        if identity_hash(features[ID]) != manifest["source_ids_sha256"][source]:
            raise ValueError("Feature population changed.")
        feature_rows += len(features)
    experiments = checked_runs(root, output, config, manifest)
    expected = {(v, f) for v in config["variant_order"] for f in config["development_folds"]}
    if set(experiments) != expected:
        raise ValueError("Every declared model is required for final verification.")
    native_differences, crossfit_reports, run_hashes = {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent/"splits"/f"{fold}.parquet")
        features, truth = load_partition(base, parent, catalog, assignment, "score", FEATURE_NAMES)
        v4, airports, sources = reference_predictions(root, config, fold, features, prior, followup)
        for name, frame in [("v4", v4), ("airports", airports)]:
            saved = pd.read_parquet(output/"references"/f"{fold}_{name}.parquet")
            pd.testing.assert_frame_equal(frame, saved, check_exact=True)
        fit_columns = ["airport", "stand", "runway", "nm_missing", "aircraft_missing", "conditional_schedule_offset_s"]
        fitting, fit_truth = load_partition(base, parent, catalog, assignment, "fit", fit_columns)
        stopping, _ = load_partition(base, parent, catalog, assignment, "early_stop", ["nm_missing"])
        first_baseline, first_map = None, None
        for variant in config["variant_order"]:
            directory, report = experiments[(variant, fold)]
            node = read_json(directory/"node.json")
            spec = config["variants"][variant]
            if node["spec"] != spec:
                raise ValueError("Native model spec differs from its declaration.")
            for frame, key in [(fitting, "fit"), (stopping, "early_stop")]:
                routed = frame.loc[route_mask(frame, spec["route"])]
                if node[key+"_rows"] != len(routed) or node[key+"_ids_sha256"] != identity_hash(routed[ID]):
                    raise ValueError("Native component population changed.")
            if node.get("baseline_path"):
                saved = pd.read_parquet(directory/"new_fit_baseline.parquet")
                full_map = read_json(directory/node["baseline_path"])
                if full_map["fit_ids_sha256"] != identity_hash(fitting[ID]) or full_map["spec"] != config["baseline"]:
                    raise ValueError("Supervised baseline used a different fitting population or recipe.")
                if not np.array_equal(saved[ID], fitting[ID]):
                    raise ValueError("Cross-fitted feature row order changed.")
                inner = group_folds(fitting, fit_truth, config["baseline"]["inner_folds"], config["baseline"]["seed"])
                np.testing.assert_array_equal(inner, saved["inner_fold"])
                for item in node["crossfit"]["folds"]:
                    held = inner == item["inner_fold"]
                    if (item["heldout_ids_sha256"] != identity_hash(fitting.loc[held, ID])
                            or item["fit_ids_sha256"] != identity_hash(fitting.loc[~held, ID])):
                        raise ValueError("Cross-fitting memberships do not reproduce.")
                if first_baseline is None:
                    first_baseline, first_map = saved, full_map
                    residual = aligned_truth(fitting, fit_truth)[TARGET].to_numpy(dtype=float)-prediction_offset(fitting, "lirf_missing_schedule")
                    hold = inner == 0
                    recreated = fit_map(fitting.loc[~hold], residual[~hold], config["baseline"])
                    indexes = np.flatnonzero(hold)[:1000]
                    observed = apply_map(fitting.iloc[indexes], recreated)
                    delta = maximum_difference(observed, saved.iloc[indexes][BASELINE_NAMES])
                    if delta > 1e-5:
                        raise ValueError("Independent cross-fitted baseline sample does not reproduce.")
                    crossfit_reports[fold] = {"rows": len(saved), "inner_folds": len(node["crossfit"]["folds"]),
                                              "recreated_inner_fold": 0, "recreated_rows": len(indexes),
                                              "max_feature_difference": delta,
                                              "assignment_sha256": object_hash(saved[[ID, "inner_fold"]].to_dict("list"))}
                else:
                    pd.testing.assert_frame_equal(saved, first_baseline, check_exact=True)
                    if full_map != first_map:
                        raise ValueError("The baseline-only comparisons must share the exact same fitting maps.")
            routed = features.loc[route_mask(features, spec["route"])].reset_index(drop=True)
            sample = sample_features(routed)
            observed = node_prediction(directory, node, sample, config["threads"])
            component = pd.read_parquet(directory/"component_predictions.parquet")
            expected_sample = sample[[ID]].merge(component, on=ID, validate="one_to_one")["prediction"].to_numpy()
            delta = maximum_difference(observed, expected_sample)
            if delta > 1e-8:
                raise ValueError("Native saved model/baseline predictions do not reproduce.")
            native_differences[variant+"/"+fold] = {"rows": len(sample), "max_difference_seconds": delta}
            values = v4["prediction"].to_numpy().copy()
            if spec["route"] == "all":
                aligned = features[[ID]].merge(component, on=ID, validate="one_to_one")["prediction"].to_numpy()
                values = .5*aligned+.5*airports["prediction"].to_numpy()
            else:
                mask = route_mask(features, spec["route"])
                values[mask] = features.loc[mask, [ID]].merge(component, on=ID, validate="one_to_one")["prediction"].to_numpy()
            saved_predictions = pd.read_parquet(directory/"predictions.parquet")
            expected_values = features[[ID]].merge(saved_predictions, on=ID, validate="one_to_one")["prediction"].to_numpy()
            if maximum_difference(values, expected_values) > 1e-8:
                raise ValueError("Candidate construction changed the saved complete predictions.")
            run_hashes[directory.name] = sha256_file(directory/"run.json")
    selection = read_json(output/"selection.json")
    rebuilt = collect(root, base, config_path, output)
    for key in ["config_sha256", "model_code_sha256", "selection_code_sha256", "bundle_id", "selected_candidate", "refit_iterations"]:
        if selection[key] != rebuilt[key]:
            raise ValueError("Selection receipt no longer reproduces: "+key)
    same_metrics(rebuilt["candidates"], selection["candidates"])
    refits = []
    for filename in sorted(output.glob("*/run.json")):
        report = read_json(filename)
        if report["stage"] != "refit":
            continue
        verify_receipts(filename.parent, report["artifacts"])
        if (report["audit_used"] or report["refit_months"] != config["refit_months"]
                or report["model_code"]["sha256"] != code["sha256"]
                or report["config_sha256"] != object_hash(config) or report["bundle_id"] != manifest["bundle_id"]
                or report["selection_sha256"] != sha256_file(output/"selection.json")):
            raise ValueError("Final refit provenance changed.")
        months = catalog[TIME].dt.strftime("%Y-%m")
        protected = set(catalog.loc[months.isin(original["descriptor"]["protocol"]["reserved_audit_months"]), "FLIGHT_ID_mvt"].dropna())
        ids = catalog.loc[months.isin(config["refit_months"]) & ~catalog["FLIGHT_ID_mvt"].isin(protected), ID]
        if report["fit_rows"] != len(ids) or report["fit_ids_sha256"] != identity_hash(ids):
            raise ValueError("Refit population differs or includes protected flight groups.")
        ranking = pd.read_parquet(base/"features/ranking.parquet")
        saved = pd.read_parquet(filename.parent/"ranking_predictions.parquet")
        if len(saved) != len(ranking) or identity_hash(saved[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Refit ranking predictions are incomplete.")
        sample = sample_features(ranking)
        values = pipeline_prediction(filename.parent, read_json(filename.parent/"pipeline.json"), sample, config["threads"])
        expected_values = sample[[ID]].merge(saved, on=ID, validate="one_to_one")["prediction"].to_numpy()
        delta = maximum_difference(values, expected_values)
        if delta > 1e-8:
            raise ValueError("The complete portable refit does not reproduce ranking predictions.")
        refits.append({"run_id": filename.parent.name, "run_sha256": sha256_file(filename),
                       "ranking_rows": len(ranking), "native_sample_rows": len(sample), "max_difference_seconds": delta})
    if require_refit and len(refits) != 1:
        raise ValueError("Exactly one verified refit is required.")
    return {"created_at_utc": utc_now(), "passed": True, "audit_used": False, "bundle_id": manifest["bundle_id"],
            "model_code_sha256": code["sha256"], "feature_rows": feature_rows, "development_runs": run_hashes,
            "native_checks": native_differences, "crossfit_checks": crossfit_reports,
            "complete_development_scores_reproduced": len(experiments), "blend_scores_reproduced": 2*selection["candidate_count"],
            "selected_candidate": selection["selected_candidate"], "selection_sha256": sha256_file(output/"selection.json"),
            "refits": refits, "verification_code_sha256": sha256_file(Path(__file__))}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase5r_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/phase5r"))
    parser.add_argument("--prior-runs", type=Path, default=Path("runs/phase5"))
    parser.add_argument("--followup-runs", type=Path, default=Path("runs/phase5_followup"))
    parser.add_argument("--output", type=Path, default=Path("docs/phase5r/verification.json"))
    parser.add_argument("--require-refit", action="store_true")
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/"data/phase5r/latest.json")["bundle"])
    report = verify(root, base, root/args.config, root/args.runs, root/args.prior_runs, root/args.followup_runs, args.require_refit)
    write_json(root/args.output, report)
    print({k: report[k] for k in ["passed", "feature_rows", "complete_development_scores_reproduced", "blend_scores_reproduced", "selected_candidate"]})


if __name__ == "__main__":
    main()
