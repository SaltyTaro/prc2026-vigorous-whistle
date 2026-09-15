"""Select only after every arrivalctx fit and every complete candidate has finished."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.validation import evaluate, identity_hash
from prc2026_arrivalctx.bundle import load_bundle, load_partition
from prc2026_ladder.inference import compose, route_mask
from prc2026_ladder.training import aligned_predictions, reference_selection, model_code_receipt as reference_code
from prc2026_arrivalctx.training import checked_runs, model_code_receipt, validate_config
from select_phase4 import paired_days


def active_variants(config, candidate):
    family = config["candidate_families"][candidate["family"]]
    return sorted({name for weights in family.values() for name in weights if name in config["variants"]})


def eligible_recipe(config, selection, manifest, code):
    if (selection["audit_used"] or selection["selected_candidate"] is None
            or selection["config_sha256"] != object_hash(config)
            or selection["bundle_id"] != manifest["bundle_id"]
            or selection["model_code_sha256"] != code["sha256"]):
        raise ValueError("An unchanged eligible selection is required before refit.")
    candidate = selection["candidates"][selection["selected_candidate"]]
    if (not candidate["eligible"] or candidate["family"] not in config["candidate_families"]
            or candidate["routes"] != config["candidate_families"][candidate["family"]]
            or candidate["new_weight"] not in config["blend_weights_new_candidate"]):
        raise ValueError("Undeclared or ineligible final recipe.")
    if set(candidate["folds"]) != set(config["development_folds"]):
        raise ValueError("Both complete folds are required.")
    for fold, row in candidate["folds"].items():
        reference = selection["reference_metrics"][fold]["overall"]["rmse_seconds"]
        score = row["metrics"]["overall"]["rmse_seconds"]
        if (not np.isfinite(score) or score/reference > 1-config["minimum_improvement_fraction_each_fold"]
                or row["metrics"]["overall"]["rows"] != selection["reference_metrics"][fold]["overall"]["rows"]):
            raise ValueError("The selected recipe fails the complete two-fold promotion margin.")
    names = active_variants(config, candidate)
    expected = {name: int(np.rint(np.median([
        selection["development_runs"][name][fold]["iterations"] for fold in config["development_folds"]
    ]))) for name in names}
    if selection["refit_iterations"] != expected or any(value < 1 for value in expected.values()):
        raise ValueError("New refit counts differ from the declared October-selected medians.")
    return candidate


def collect(root, base, config_path, output):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(root, config, original["descriptor"]["protocol"])
    if sha256_file(root/"configs/features_arrivalctx_v1.json") != config["feature_configuration_sha256"]:
        raise ValueError("The new feature declaration changed.")
    prior = reference_selection(root, config)
    matched = read_json(root/config["matched_reference_selection"])
    experiments = checked_runs(root, output, config, manifest)
    expected = {(name, fold) for name in config["variant_order"] for fold in config["development_folds"]}
    if set(experiments) != expected:
        raise ValueError("All four declared runs must complete before any selection.")
    truths, features, references, components, ref_metrics, runs = {}, {}, {}, {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent/"splits"/(fold+".parquet"))
        features[fold], truths[fold] = load_partition(
            base, parent, catalog, assignment, "score", ["nmclock_available", "nmclock_proxy_s"])
        ref_dir = output/"references"/fold
        receipt = read_json(ref_dir/"reference.json")
        if receipt["source_run_sha256"] != prior["development_runs"][config["reference_variant"]][fold]["run_sha256"]:
            raise ValueError("Saved reference differs from frozen v6.")
        if receipt["config_sha256"] != object_hash(config) or receipt["model_code_sha256"] != reference_code(root)["sha256"]:
            raise ValueError("Reference configuration or producing code changed.")
        verify_receipts(ref_dir, receipt["artifacts"])
        references[fold] = aligned_predictions(features[fold], pd.read_parquet(ref_dir/"v6.parquet"))
        ref_metrics[fold] = evaluate(truths[fold], references[fold], original["ranking_month_weights"] if fold == "seasonal" else None)
        expected_rmse = prior["candidates"][prior["selected_candidate"]]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
        if not np.isclose(ref_metrics[fold]["overall"]["rmse_seconds"], expected_rmse, rtol=1e-12, atol=1e-9):
            raise ValueError("The complete v6 reference score changed.")
        components[fold] = {}
        for name in ["v6_global", "v6_airports"]:
            components[fold][name] = aligned_predictions(
                features[fold], pd.read_parquet(ref_dir/(name+".parquet")))["prediction"].to_numpy()
        for name in config["variant_order"]:
            path, report = experiments[(name, fold)]
            for role in ["fit", "early_stop", "score"]:
                if report[role+"_ids_sha256"] != original["folds"][fold]["identity_sha256"][role]:
                    raise ValueError("Protected fold identities changed.")
            prediction = aligned_predictions(features[fold], pd.read_parquet(path/"predictions.parquet"))
            metrics = evaluate(truths[fold], prediction, original["ranking_month_weights"] if fold == "seasonal" else None)
            saved = read_json(path/"metrics.json")
            if not np.isclose(metrics["overall"]["rmse_seconds"], saved["overall"]["rmse_seconds"], rtol=1e-12, atol=1e-9):
                raise ValueError("A complete native-model score no longer reproduces.")
            components[fold][name] = prediction["prediction"].to_numpy()
            runs.setdefault(name, {})[fold] = {"run_id": path.name, "run_sha256": sha256_file(path/"run.json"),
                                               "iterations": report["best_iterations"], "metrics": metrics,
                                               "resources": report["resources"]}
    candidates = {}
    for family_name, family in config["candidate_families"].items():
        for weight in config["blend_weights_new_candidate"]:
            name = f"{family_name}_v6_{int(round(100*weight))}"
            folds, ratios = {}, []
            for fold in config["development_folds"]:
                values = compose(references[fold]["prediction"], components[fold],
                                 route_mask(features[fold], "clock_available"), family, weight)
                prediction = pd.DataFrame({ID: features[fold][ID], "prediction": values})
                metrics = evaluate(truths[fold], prediction, original["ranking_month_weights"] if fold == "seasonal" else None)
                ratio = metrics["overall"]["rmse_seconds"]/ref_metrics[fold]["overall"]["rmse_seconds"]
                ratios.append(ratio)
                folds[fold] = {"metrics": metrics, "rmse_ratio_to_v6": ratio}
            candidates[name] = {"family": family_name, "routes": family, "new_weight": weight, "folds": folds,
                                "mean_relative_rmse": float(np.mean(ratios)),
                                "eligible": all(ratio <= 1-config["minimum_improvement_fraction_each_fold"] for ratio in ratios)}
    eligible = [name for name, candidate in candidates.items() if candidate["eligible"]]
    selected = min(eligible, key=lambda name: (candidates[name]["mean_relative_rmse"], name)) if eligible else None
    iterations, paired = {}, {}
    if selected:
        candidate = candidates[selected]
        iterations = {name: int(np.rint(np.median([
            runs[name][fold]["iterations"] for fold in config["development_folds"]
        ]))) for name in active_variants(config, candidate)}
        for fold in config["development_folds"]:
            values = compose(references[fold]["prediction"], components[fold],
                             route_mask(features[fold], "clock_available"), candidate["routes"], candidate["new_weight"])
            prediction = pd.DataFrame({ID: features[fold][ID], "prediction": values})
            paired[fold] = paired_days(truths[fold], prediction, references[fold], config["seed"])
            paired[fold]["comparison"] = "selected candidate minus frozen v6"
    return {"created_at_utc": utc_now(), "configuration": config, "config_sha256": object_hash(config),
            "model_code_sha256": model_code_receipt(root)["sha256"], "selection_code_sha256": sha256_file(Path(__file__)),
            "paired_day_code_sha256": sha256_file(root/"tools/select_phase4.py"),
            "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"], "audit_used": False,
            "reference_selection_sha256": config["reference_selection_sha256"], "reference_metrics": ref_metrics,
            "matched_reference_metrics": {name: {fold: matched["development_runs"][variant["matched_reference_variant"]][fold]["metrics"] for fold in config["development_folds"]} for name, variant in config["variants"].items()},
            "matched_reference_selection_sha256": config["matched_reference_selection_sha256"],
            "development_runs": runs, "candidate_count": len(candidates), "candidates": candidates,
            "selected_candidate": selected, "refit_iterations": iterations, "selected_paired_days": paired,
            "refit_months": config["refit_months"], "rule": config["selection"]}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/arrivalctx_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/arrivalctx"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/arrivalctx"))
    args = parser.parse_args()
    destinations = [root/args.runs/"selection.json", root/args.report_dir/"development_results.json"]
    if any(path.exists() for path in destinations):
        raise ValueError("Preserve the existing selection receipt; independent verification can recompute it.")
    base = args.bundle or Path(read_json(root/"data/arrivalctx/latest.json")["bundle"])
    report = collect(root, root/base, root/args.config, root/args.runs)
    for path in destinations:
        write_json(path, report)
    leading = sorted(report["candidates"].items(), key=lambda item: item[1]["mean_relative_rmse"])[:5]
    print({"selected": report["selected_candidate"], "candidate_count": report["candidate_count"],
           "refit_iterations": report["refit_iterations"],
           "leading_candidates": [{"name": name, "eligible": row["eligible"],
              **{fold: row["folds"][fold]["metrics"]["overall"]["rmse_seconds"] for fold in ["seasonal", "forward"]}}
             for name, row in leading]})


if __name__ == "__main__":
    main()
