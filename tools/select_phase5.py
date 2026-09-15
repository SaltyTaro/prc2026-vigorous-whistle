"""Rescore the bounded batch, freeze a seed plan, then select a declared blend."""

import argparse
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID
from prc2026.phase1 import load_partition as original_partition
from prc2026.validation import evaluate
from prc2026_phase4.bundle import load_bundle
from prc2026_phase5.inference import combine_predictions, validate_weights
from prc2026_phase5.training import checked_runs, model_code_receipt, reference_runs, validate_config
from select_phase4 import paired_days


def declared_weights(config, initial_reports, anchor, followup=None):
    """A small common weight grid; no per-airport or per-fold coefficient fitting."""
    names = config["variant_order"] + [f"seed_{s}" for s in config["seed_replicas"]]
    recipes = {name: {name: 1.0} for name in names}
    trio = [anchor, *[f"seed_{s}" for s in config["seed_replicas"]]]
    average = {name: 1/3 for name in trio}
    recipes["three_seed_mean"] = average
    alternatives = sorted((n for n in config["variant_order"] if n != "reference"),
                          key=lambda n: (initial_reports[n]["mean_relative_rmse"], n))[:config["blend_policy"]["top_initial_alternatives"]]
    for left, right in combinations(["reference", *alternatives], 2):
        for weight in config["blend_policy"]["weights"]:
            recipes[f"blend_{left}_{right}_{int(weight*100)}"] = {left: weight, right: 1-weight}
    for weight in config["blend_policy"]["weights"]:
        mix = {name: (1-weight)*value for name, value in average.items()}
        mix["reference"] = mix.get("reference", 0.0) + weight
        recipes[f"reference_seed_mean_{int(weight*100)}"] = mix
    if followup is not None:
        extension = followup["blend_extension"]
        if (extension["pair_with"] != ["reference", "three_seed_mean"]
                or not extension["include_followup_individual"]
                or not extension["weights_shared_across_folds_and_airports"]
                or not extension["promotion_margin_unchanged"]):
            raise ValueError("Unknown follow-up blend policy.")
        for name in followup["variant_order"]:
            if name == "reference":
                continue
            recipes[name] = {name: 1.0}
            for other, members in [("reference", {"reference": 1.0}), ("three_seed_mean", average)]:
                for weight in extension["weights"]:
                    mix = {n: (1-weight)*v for n, v in members.items()}
                    mix[name] = weight
                    recipes[f"followup_{name}_{other}_{int(weight*100)}"] = mix
    unique, seen = {}, set()
    for name, weights in recipes.items():
        validate_weights(list(weights.values()))
        identity = object_hash({k: round(v, 12) for k, v in weights.items()})
        if identity not in seen:
            unique[name] = weights
            seen.add(identity)
    return unique


def common_structure(first, second):
    if (first["default"] != second["default"] or first["patches"] != second["patches"]
            or set(first["nodes"]) != set(second["nodes"])):
        raise ValueError("The two fold model structures differ.")
    for name, node in first["nodes"].items():
        other = second["nodes"][name]
        if any(node[k] != other[k] for k in ["schema", "offset", "route", "settings"]):
            raise ValueError("The two fold component settings differ.")
        if node.get("model_kind", "lightgbm") != other.get("model_kind", "lightgbm"):
            raise ValueError("The two fold model families differ.")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["initial", "final"], required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs/phase5"))
    parser.add_argument("--reference-runs", type=Path, default=Path("runs/phase4"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_v1.json"))
    parser.add_argument("--bundle", type=Path, default=Path("data/phase4/2ceb3449c26767a97311"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/phase5"))
    parser.add_argument("--followup-runs", type=Path)
    parser.add_argument("--followup-config", type=Path, default=Path("configs/phase5_followup_v1.json"))
    args = parser.parse_args()
    output, docs = root / args.runs, root / args.report_dir
    config = read_json(root / args.config)
    manifest, parent, original, catalog = load_bundle(root, root / args.bundle)
    validate_config(config, original["descriptor"]["protocol"])
    references = reference_runs(root, root / args.reference_runs, config, original)
    experiments = checked_runs(root, output, config, manifest)
    variants = config["variant_order"] + ([] if args.mode == "initial" else [f"seed_{s}" for s in config["seed_replicas"]])
    expected = {(v, f) for v in variants for f in config["development_folds"]}
    if not expected <= set(experiments):
        raise ValueError("Every declared run must complete before this selection step.")
    if set(experiments) - {(v, f) for v in config["variant_order"] + [f"seed_{s}" for s in config["seed_replicas"]] for f in config["development_folds"]}:
        raise ValueError("Undeclared development run found.")
    followup, followup_references = None, {}
    if args.followup_runs:
        if args.mode != "final":
            raise ValueError("The original seed anchor must be selected before incorporating the follow-up.")
        followup = read_json(root / args.followup_config)
        validate_config(followup, original["descriptor"]["protocol"])
        if followup["adaptive_followup"]["parent_configuration_sha256"] != object_hash(config):
            raise ValueError("Follow-up declaration has the wrong original batch.")
        for key in ["feature_bundle_id", "development_folds", "refit_months", "seed", "threads", "minimum_improvement_fraction_each_fold"]:
            if followup[key] != config[key]:
                raise ValueError("Follow-up changed a frozen experiment policy.")
        additional = checked_runs(root, root / args.followup_runs, followup, manifest)
        if set(additional) != {(v, f) for v in followup["variant_order"] for f in config["development_folds"]}:
            raise ValueError("Every declared follow-up run must complete, without additional experiments.")
        for key, value in additional.items():
            if key[0] == "reference":
                followup_references[key[1]] = value
            elif key in experiments:
                raise ValueError("Follow-up variant name is ambiguous.")
            else:
                experiments[key] = value
        variants += [v for v in followup["variant_order"] if v != "reference"]
    truth, reference_predictions, reference_metrics, predictions, reports = {}, {}, {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent / "splits" / f"{fold}.parquet")
        _, truth[fold] = original_partition(parent, catalog, assignment, "score")
        reference_predictions[fold] = pd.read_parquet(references[fold][0] / "predictions.parquet")
        reference_metrics[fold] = evaluate(truth[fold], reference_predictions[fold], original["ranking_month_weights"] if fold == "seasonal" else None)
        if fold in followup_references:
            observed = pd.read_parquet(followup_references[fold][0] / "predictions.parquet").sort_values(ID).reset_index(drop=True)
            expected_reference = reference_predictions[fold].sort_values(ID).reset_index(drop=True)
            pd.testing.assert_frame_equal(observed, expected_reference, check_exact=True)
    for variant in variants:
        folds, ratios = {}, []
        for fold in config["development_folds"]:
            path, run = experiments[(variant, fold)]
            for role in ["fit", "early_stop", "score"]:
                if run[role + "_ids_sha256"] != original["folds"][fold]["identity_sha256"][role]:
                    raise ValueError("Protected split IDs changed.")
            predictions[(variant, fold)] = pd.read_parquet(path / "predictions.parquet")
            metrics = evaluate(truth[fold], predictions[(variant, fold)], original["ranking_month_weights"] if fold == "seasonal" else None)
            if not np.isclose(metrics["overall"]["rmse_seconds"], read_json(path / "metrics.json")["overall"]["rmse_seconds"], rtol=1e-12):
                raise ValueError("Saved development score does not reproduce.")
            ratio = metrics["overall"]["rmse_seconds"] / reference_metrics[fold]["overall"]["rmse_seconds"]
            ratios.append(ratio)
            folds[fold] = {"run_id": run["run_id"], "run_sha256": sha256_file(path / "run.json"),
                           "run_group": "followup" if followup is not None and variant in followup["variants"] and variant != "reference" else "initial",
                           "metrics": metrics, "rmse_ratio_to_v3": ratio, "resources": run["resources"],
                           "component_iterations": run["component_iterations"]}
        reports[variant] = {"folds": folds, "mean_relative_rmse": float(np.mean(ratios)), "improves_both": all(r < 1 for r in ratios)}
    eligible_anchors = ["reference", *[v for v in config["variant_order"] if v != "reference" and reports[v]["improves_both"]]]
    anchor = min(eligible_anchors, key=lambda v: reports[v]["mean_relative_rmse"])
    common = {"created_at_utc": utc_now(), "config_sha256": object_hash(config),
              "model_code_sha256": model_code_receipt(root)["sha256"], "selection_code_sha256": sha256_file(Path(__file__)),
              "paired_day_utility_sha256": sha256_file(root / "tools/select_phase4.py"),
              "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"], "audit_used": False}
    if args.mode == "initial":
        plan = {**common, "anchor_variant": anchor, "anchor_runs": reports[anchor]["folds"],
                "seed_replicas": config["seed_replicas"], "rule": config["seed_anchor_rule"]}
        write_json(output / "seed_plan.json", plan)
        write_json(docs / "initial_results.json", {**common, "configuration": config, "reference_metrics": reference_metrics,
                                                   "candidates": reports, "seed_plan": plan})
        print("Seed anchor:", anchor)
    else:
        plan = read_json(output / "seed_plan.json")
        if plan["anchor_variant"] != anchor or plan["config_sha256"] != common["config_sha256"] or plan["model_code_sha256"] != common["model_code_sha256"]:
            raise ValueError("The seed selection recipe changed after replicas were fitted.")
        for seed in config["seed_replicas"]:
            for fold in config["development_folds"]:
                run = experiments[(f"seed_{seed}", fold)][1]
                if run["seed_override"] != seed or run["source_run"]["run_sha256"] != plan["anchor_runs"][fold]["run_sha256"]:
                    raise ValueError("Seed run is not a replica of the frozen anchor.")
        candidates, airport_rows = {}, []
        recipes = declared_weights(config, reports, anchor, followup)
        margin = config["minimum_improvement_fraction_each_fold"]
        for candidate, weights in recipes.items():
            folds, ratios = {}, []
            for fold in config["development_folds"]:
                combined = combine_predictions({v: predictions[(v, fold)] for v in weights}, weights)
                metrics = evaluate(truth[fold], combined, original["ranking_month_weights"] if fold == "seasonal" else None)
                ratio = metrics["overall"]["rmse_seconds"] / reference_metrics[fold]["overall"]["rmse_seconds"]
                ratios.append(ratio)
                path = output / "selection_predictions" / f"{candidate}_{fold}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    pd.testing.assert_frame_equal(pd.read_parquet(path), combined, check_exact=True)
                else:
                    combined.to_parquet(path, index=False, compression="zstd")
                uncertainty = paired_days(truth[fold], combined, reference_predictions[fold], config["seed"])
                uncertainty["comparison"] = "candidate minus Phase-4 v3 blend"
                folds[fold] = {"metrics": metrics, "rmse_ratio_to_v3": ratio, "paired_day_uncertainty": uncertainty,
                               "prediction_artifact": {"path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}}
                for airport, metric in metrics["by_airport"].items():
                    airport_rows.append({"candidate": candidate, "fold": fold, "airport": airport, **metric})
            candidates[candidate] = {"weights": weights, "folds": folds, "mean_relative_rmse": float(np.mean(ratios)),
                                     "eligible": all(r <= 1-margin for r in ratios)}
        eligible = [n for n, c in candidates.items() if c["eligible"]]
        chosen = min(eligible, key=lambda n: candidates[n]["mean_relative_rmse"]) if eligible else None
        pipeline, iterations = None, {}
        if chosen:
            members = []
            for name, weight in candidates[chosen]["weights"].items():
                pipes = [read_json(experiments[(name, fold)][0] / "pipeline.json") for fold in config["development_folds"]]
                common_structure(*pipes)
                for component in pipes[0]["nodes"]:
                    iterations[name + "/" + component] = int(np.rint(np.median([p["nodes"][component]["iterations"] for p in pipes])))
                members.append({"name": name, "weight": weight, "pipeline": deepcopy(pipes[0])})
            pipeline = {"kind": "mixture", "members": members}
        selection = {**common, "configuration": config, "selected_candidate": chosen, "rule": config["selection"],
                     "followup_configuration": followup, "followup_configuration_sha256": None if followup is None else object_hash(followup),
                     "seed_plan_sha256": sha256_file(output / "seed_plan.json"), "initial_and_seed_runs": reports,
                     "reference_metrics": reference_metrics, "candidates": candidates,
                     "refit_pipeline": pipeline, "refit_component_iterations": iterations,
                     "refit_months": config["refit_months"], "candidate_count": len(candidates)}
        write_json(output / "selection.json", selection)
        write_json(docs / "development_results.json", selection)
        pd.DataFrame(airport_rows).to_csv(docs / "airport_errors.csv", index=False)
        print("Selected:", chosen, "from", len(candidates), "declared candidates")
        if chosen:
            print("Weights:", candidates[chosen]["weights"])
            print("Selected RMSE:", {f: d["metrics"]["overall"]["rmse_seconds"] for f, d in candidates[chosen]["folds"].items()})
    for variant, report in reports.items():
        print(variant, {f: round(v["metrics"]["overall"]["rmse_seconds"], 6) for f, v in report["folds"].items()})


if __name__ == "__main__":
    main()
