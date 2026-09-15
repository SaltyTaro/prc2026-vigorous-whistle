"""Rescore the complete declared batch and freeze a v5-relative promotion decision."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID
from prc2026.phase1 import load_partition as original_partition
from prc2026.validation import evaluate, identity_hash
from prc2026_phase5.inference import combine_predictions
from prc2026_nmclock.bundle import load_bundle
from prc2026_nmclock.training import checked_runs, model_code_receipt, reference_selection, validate_config
from select_phase4 import paired_days
from prc2026_phase3.bundle import load_partition
from prc2026_nmclock.features import route_mask


def collect(root: Path, base: Path, config_path: Path, output: Path) -> dict:
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    old_selection = reference_selection(root, config)
    experiments = checked_runs(root, output, config, manifest)
    expected = {(v, f) for v in config["variant_order"] for f in config["development_folds"]}
    if set(experiments) != expected:
        raise ValueError("Every declared experiment must finish before selection, without additional runs.")
    truths, references, ref_metrics, predictions, runs, missing_ids = {}, {}, {}, {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent/"splits"/f"{fold}.parquet")
        _, truths[fold] = original_partition(parent, catalog, assignment, "score")
        routed, _ = load_partition(base, parent, catalog, assignment, "score", ["nmclock_available", "nmclock_proxy_s"])
        missing_ids[fold] = set(routed.loc[~route_mask(routed), ID])
        ref_dir = output/"references"
        receipt = read_json(ref_dir/f"{fold}.json")
        for kind in ["v5"]:
            if sha256_file(ref_dir/f"{fold}_{kind}.parquet") != receipt[kind+"_sha256"]:
                raise ValueError("Reference prediction receipt changed.")
        if receipt["reference_selection_sha256"] != config["reference_selection_sha256"]:
            raise ValueError("Reference selection identity changed.")
        references[fold] = pd.read_parquet(ref_dir/f"{fold}_v5.parquet")
        ref_metrics[fold] = evaluate(truths[fold], references[fold], original["ranking_month_weights"] if fold == "seasonal" else None)
        expected_rmse = old_selection["candidates"][old_selection["selected_candidate"]]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
        if not np.isclose(ref_metrics[fold]["overall"]["rmse_seconds"], expected_rmse, rtol=1e-12, atol=1e-9):
            raise ValueError("Frozen v5 reference score changed.")
        for variant in config["variant_order"]:
            path, run = experiments[(variant, fold)]
            for role in ["fit", "early_stop", "score"]:
                if run[role+"_ids_sha256"] != original["folds"][fold]["identity_sha256"][role]:
                    raise ValueError("Protected fitting/stopping/scoring IDs changed.")
            predictions[(variant, fold)] = pd.read_parquet(path/"predictions.parquet")
            if identity_hash(predictions[(variant, fold)][ID]) != original["folds"][fold]["identity_sha256"]["score"]:
                raise ValueError("Incomplete scoring predictions.")
            metrics = evaluate(truths[fold], predictions[(variant, fold)], original["ranking_month_weights"] if fold == "seasonal" else None)
            saved = read_json(path/"metrics.json")
            if not np.isclose(metrics["overall"]["rmse_seconds"], saved["overall"]["rmse_seconds"], rtol=1e-12, atol=1e-9):
                raise ValueError("A saved development score does not reproduce.")
            runs.setdefault(variant, {})[fold] = {"run_id": path.name, "run_sha256": sha256_file(path/"run.json"),
                                                  "iterations": run["best_iterations"], "component_iterations": run["component_iterations"], "metrics": metrics,
                                                  "resources": run["resources"]}
    candidates = {}
    for variant, family_weights in config["candidate_families"].items():
        family_predictions = {f: combine_predictions({v: predictions[(v, f)] for v in family_weights}, family_weights) for f in config["development_folds"]}
        for weight in config["blend_weights_new_candidate"]:
            name = f"{variant}_v5_{int(round(weight*100))}"
            folds, ratios = {}, []
            for fold in config["development_folds"]:
                mixed = combine_predictions({"candidate": family_predictions[fold], "v5": references[fold]},
                                            {"candidate": weight, "v5": 1-weight})
                absent = mixed[ID].isin(missing_ids[fold]).to_numpy()
                baseline = mixed[[ID]].merge(references[fold], on=ID, how='left', validate='one_to_one')['prediction'].to_numpy()
                mixed.loc[absent, 'prediction'] = baseline[absent]
                metric = evaluate(truths[fold], mixed, original["ranking_month_weights"] if fold == "seasonal" else None)
                ratio = metric["overall"]["rmse_seconds"]/ref_metrics[fold]["overall"]["rmse_seconds"]
                ratios.append(ratio)
                interval = paired_days(truths[fold], mixed, references[fold], config["seed"])
                interval["comparison"] = "candidate minus frozen v5"
                folds[fold] = {"metrics": metric, "rmse_ratio_to_v5": ratio, "paired_days": interval}
            candidates[name] = {"variant": variant, "family_weights": family_weights, "new_weight": weight, "folds": folds,
                                "mean_relative_rmse": float(np.mean(ratios)),
                                "eligible": all(r <= 1-config["minimum_improvement_fraction_each_fold"] for r in ratios)}
    eligible = [name for name, report in candidates.items() if report["eligible"]]
    selected = min(eligible, key=lambda name: (candidates[name]["mean_relative_rmse"], name)) if eligible else None
    iterations = {}
    if selected:
        for variant in candidates[selected]['family_weights']:
            cards = [runs[variant][f]['component_iterations'] for f in config['development_folds']]
            iterations[variant] = {'global': int(np.rint(np.median([c['global'] for c in cards]))),
                'airports': {a: int(np.rint(np.median([c['airports'][a] for c in cards]))) for a in config['airports']}}
    return {"created_at_utc": utc_now(), "configuration": config, "config_sha256": object_hash(config),
            "model_code_sha256": model_code_receipt(root)["sha256"], "selection_code_sha256": sha256_file(Path(__file__)),
            "paired_day_code_sha256": sha256_file(root/"tools/select_phase4.py"),
            "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
            "audit_used": False, "reference_selection_sha256": config["reference_selection_sha256"],
            "reference_metrics": ref_metrics, "development_runs": runs, "candidate_count": len(candidates),
            "candidates": candidates, "selected_candidate": selected, "refit_iterations": iterations,
            "refit_months": config["refit_months"], "rule": config["selection"]}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/nmclock_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/nmclock"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/nmclock"))
    args = parser.parse_args()
    base = root/(args.bundle or read_json(root/"data/nmclock/latest.json")["bundle"])
    report = collect(root, base, root/args.config, root/args.runs)
    write_json(root/args.runs/"selection.json", report)
    write_json(root/args.report_dir/"development_results.json", report)
    for name, row in report["candidates"].items():
        print(name, *(f"{f}={row['folds'][f]['metrics']['overall']['rmse_seconds']:.6f}" for f in ["seasonal", "forward"]), "eligible="+str(row["eligible"]))
    print("Selected:", report["selected_candidate"], "New-model refit trees:", report["refit_iterations"])


if __name__ == "__main__":
    main()
