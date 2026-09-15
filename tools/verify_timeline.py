"""Verify complete timeline results, independent features and native predictions."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.phase1 import validate_inputs
from prc2026.validation import identity_hash
from prc2026_models.experiments import load_model, predict
from prc2026_phase3.bundle import load_partition
from prc2026_nmclock.features import route_mask
from prc2026_nmclock.inference import node_prediction as parent_node, pipeline_prediction as parent_prediction
from prc2026_ladder.training import aligned_predictions, reference_selection
from prc2026_timeline.bundle import load_bundle
from prc2026_timeline.features import FEATURE_NAMES, CATEGORICAL_COLUMNS
from prc2026_timeline.encoding import contract, transform
from prc2026_timeline.inference import node_prediction
from prc2026_timeline.training import checked_runs, model_code_receipt, validate_config
from check_timeline_features import verify_feature_samples
from select_timeline import collect
from verify_phase5 import same_metrics


def difference(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Native predictions need matching finite populations.")
    delta = float(np.max(np.abs(left-right))) if len(left) else 0.
    if delta > 1e-8:
        raise ValueError("Native prediction reconstruction differs.")
    return delta


def features_receipt(root, base, manifest, original, destination):
    expected = {"bundle_id": manifest["bundle_id"], "manifest_sha256": sha256_file(base/"manifest.json"),
                "feature_oracle_code_sha256": sha256_file(root/"tools/check_timeline_features.py"),
                "verifier_code_sha256": sha256_file(Path(__file__))}
    if destination.exists():
        prior = read_json(destination)
        if not prior["passed"] or any(prior[key] != value for key, value in expected.items()):
            raise ValueError("Previous feature verification has different inputs or code.")
        return prior
    result = verify_feature_samples(root, base, manifest, original)
    report = {"created_at_utc": utc_now(), "passed": True, **expected, **result}
    write_json(destination, report)
    return report


def native_sample(features):
    routed = features.loc[route_mask(features)].reset_index(drop=True)
    special = (routed.nmclock_proxy_s.lt(0) | routed.nmclock_proxy_s.gt(10800)
               | routed.nmclock_clock_difference_s.abs().gt(3600) | routed.flight_arrival_before_movement.eq(1))
    indexes = np.unique(np.r_[np.linspace(0, len(routed)-1, min(1000, len(routed)), dtype=int),
                             np.flatnonzero(special), routed.groupby("airport", sort=True).head(1).index])
    return routed.iloc[indexes].reset_index(drop=True)


def verify(root, base, config_path, output, previous_runs, feature_report):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    if validate_inputs(root) != manifest["descriptor"]["input_files"]:
        raise ValueError("Supplied input files changed.")
    feature_checks = features_receipt(root, base, manifest, original, feature_report)
    saved = read_json(output/"selection.json")
    rebuilt = collect(root, base, config_path, output)
    for key in ["config_sha256", "model_code_sha256", "selection_code_sha256", "paired_day_code_sha256", "bundle_id", "parent_bundle_id", "candidate_count", "selected_candidate", "refit_iterations"]:
        if saved[key] != rebuilt[key]:
            raise ValueError("Timeline selection changed: "+key)
    for key in ["candidates", "reference_metrics", "matched_lgb_reference_metrics", "selected_paired_days"]:
        same_metrics(rebuilt[key], saved[key])
    experiments = checked_runs(root, output, config, manifest)
    prior = reference_selection(root, config)
    native, complete_scores, references, hashes = {}, {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent/"splits"/f"{fold}.parquet")
        scoring, _ = load_partition(base, parent, catalog, assignment, "score", FEATURE_NAMES)
        fitting, _ = load_partition(base, parent, catalog, assignment, "fit", list(dict.fromkeys(["nmclock_available", "nmclock_proxy_s", *CATEGORICAL_COLUMNS])))
        stopping, _ = load_partition(base, parent, catalog, assignment, "early_stop", ["nmclock_available", "nmclock_proxy_s"])
        fit = fitting.loc[route_mask(fitting)]
        stop = stopping.loc[route_mask(stopping)]
        vocab = {name: sorted(set(fit[name].astype("string").fillna("__MISSING__")) | {"__MISSING__"}) for name in CATEGORICAL_COLUMNS}
        chosen = native_sample(scoring)
        full_ref = aligned_predictions(scoring, pd.read_parquet(output/"references"/fold/"v6.parquet"))
        card = prior["development_runs"][config["reference_variant"]][fold]
        source = previous_runs/card["run_id"]
        if sha256_file(source/"run.json") != card["run_sha256"]:
            raise ValueError("Frozen v6 run changed.")
        verify_receipts(source, read_json(source/"run.json")["artifacts"])
        old_nodes = read_json(source/"nodes.json")
        global_values = parent_node(source, old_nodes["global"], chosen, config["threads"])
        airport_values = parent_prediction(source, {"kind": "nmclock_airports", "nodes": old_nodes["airports"]}, chosen, config["threads"])
        expected_ref = chosen[[ID]].merge(full_ref, on=ID, validate="one_to_one")["prediction"].to_numpy()
        ref_delta = difference(.5*global_values+.5*airport_values, expected_ref)
        saved_airports = pd.read_parquet(output/"references"/fold/"v6_airports.parquet")
        airport_delta = difference(airport_values, chosen[[ID]].merge(saved_airports, on=ID, validate="one_to_one")["prediction"])
        references[fold] = {"rows": len(chosen), "maximum_v6_difference_seconds": ref_delta, "maximum_airport_difference_seconds": airport_delta}
        missing = ~route_mask(scoring)
        native[fold], complete_scores[fold], hashes[fold] = {}, {}, {}
        for name, spec in config["variants"].items():
            run, report = experiments[(name, fold)]
            node = read_json(run/"node.json")
            settings = {key: config[key] for key in ["threads", "seed", "lightgbm", "catboost"]}
            if node["schema"] != "timeline" or node["spec"] != spec or node["settings"] != settings or node["target_filter"] != "none":
                raise ValueError("A native model changed its feature, fitting or model contract.")
            if node["fit_rows"] != len(fit) or node["fit_ids_sha256"] != identity_hash(fit[ID]) or node["early_stop_rows"] != len(stop) or node["early_stop_ids_sha256"] != identity_hash(stop[ID]):
                raise ValueError("Native fitting or October stopping population changed.")
            if not 1 <= node["iterations"] <= config[spec["kind"]]["rounds"] or node["iterations"] != report["best_iterations"]:
                raise ValueError("Invalid selected native iteration count.")
            encoder = read_json(run/node["encoder_path"])
            names, cats = contract(spec)
            if encoder["feature_names"] != names or encoder["categorical_columns"] != cats or encoder["vocabularies"] != vocab:
                raise ValueError("Categorical encoders must use the fitting route only.")
            expected = chosen[[ID]].merge(pd.read_parquet(run/"predictions.parquet"), on=ID, validate="one_to_one")["prediction"].to_numpy()
            actual = node_prediction(run, node, chosen, config["threads"])
            direct = predict(load_model(run/node["model_path"], spec["kind"]), transform(chosen, encoder), spec["kind"], config["threads"])+chosen.nmclock_proxy_s.to_numpy(dtype=float)
            delta = max(difference(actual, expected), difference(direct, expected))
            complete = aligned_predictions(scoring, pd.read_parquet(run/"predictions.parquet"))
            if not np.array_equal(complete.loc[missing, "prediction"].to_numpy(), full_ref.loc[missing, "prediction"].to_numpy()):
                raise ValueError("A missing-clock fallback changed.")
            native[fold][name] = {"rows": len(chosen), "maximum_difference_seconds": delta, "complete_unchanged_fallback_rows": int(missing.sum()), "fit_encoder_verified": True}
            complete_scores[fold][name] = rebuilt["development_runs"][name][fold]["metrics"]["overall"]["rmse_seconds"]
            hashes[fold][name] = sha256_file(run/"run.json")
    return {"created_at_utc": utc_now(), "passed": True, "audit_used": False, "bundle_id": manifest["bundle_id"],
            "model_code_sha256": model_code_receipt(root)["sha256"], "selection_sha256": sha256_file(output/"selection.json"),
            "verifier_code_sha256": sha256_file(Path(__file__)), "feature_checks": feature_checks,
            "native_prediction_checks": native, "complete_scores": complete_scores, "reference_checks": references,
            "run_sha256": hashes, "development_run_count": len(experiments), "candidate_count": saved["candidate_count"],
            "selected_candidate": saved["selected_candidate"], "refit_iterations": saved["refit_iterations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/timeline_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/timeline"))
    parser.add_argument("--previous-runs", type=Path, default=Path("runs/nmclock"))
    parser.add_argument("--output", type=Path, default=Path("docs/timeline/verification.json"))
    parser.add_argument("--feature-report", type=Path, default=Path("docs/timeline/feature_verification.json"))
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    base = ROOT/(args.bundle or Path(read_json(ROOT/"data/timeline/latest.json")["bundle"]))
    if args.features_only:
        manifest, _, original, _ = load_bundle(ROOT, base)
        if validate_inputs(ROOT) != manifest["descriptor"]["input_files"]:
            raise ValueError("Supplied inputs changed.")
        report = features_receipt(ROOT, base, manifest, original, ROOT/args.feature_report)
    else:
        if (ROOT/args.output).exists():
            raise ValueError("Preserve existing timeline verification reports.")
        report = verify(ROOT, base, ROOT/args.config, ROOT/args.runs, ROOT/args.previous_runs, ROOT/args.feature_report)
        write_json(ROOT/args.output, report)
    print({key: report[key] for key in ["passed", "bundle_id"]})


if __name__ == "__main__":
    main()
