"""Independently verify frozen populations, native models, and complete blends."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TIME
from prc2026.phase1 import load_partition as original_partition
from prc2026.validation import evaluate, identity_hash
from prc2026_phase4.bundle import load_bundle, scientific_parent
from prc2026_phase4.features import FEATURE_NAMES, route_mask
from prc2026_phase5.inference import iter_nodes, pipeline_prediction, validate_weights
from prc2026_phase5.training import checked_runs, derive_settings, model_code_receipt, reference_runs, validate_config
from select_phase5 import declared_weights


def same_metrics(observed, expected, location="metrics"):
    """Check every reported slice, allowing only floating-point summation noise."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise ValueError(f"Metric type differs at {location}.")
        # JSON object keys are strings; evaluate() retains integer month keys.
        # Normalize only this representation and reject key collisions.
        original_count = len(observed)
        observed = {str(key): value for key, value in observed.items()}
        if len(observed) != original_count or set(observed) != set(expected):
            raise ValueError(f"Metric keys differ at {location}.")
        for key in expected:
            same_metrics(observed[key], expected[key], location + "/" + str(key))
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise ValueError(f"Metric lengths differ at {location}.")
        for index, (left, right) in enumerate(zip(observed, expected)):
            same_metrics(left, right, location + "/" + str(index))
    elif isinstance(expected, float):
        if not np.isclose(observed, expected, rtol=1e-11, atol=1e-7):
            raise ValueError(f"Metric differs at {location}.")
    elif observed != expected:
        raise ValueError(f"Metric differs at {location}.")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/phase5"))
    parser.add_argument("--followup-runs", type=Path, default=Path("runs/phase5_followup"))
    parser.add_argument("--reference-runs", type=Path, default=Path("runs/phase4"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase5_v1.json"))
    parser.add_argument("--followup-config", type=Path, default=Path("configs/phase5_followup_v1.json"))
    parser.add_argument("--bundle", type=Path, default=Path("data/phase4/2ceb3449c26767a97311"))
    parser.add_argument("--output", type=Path, default=Path("docs/phase5/verification.json"))
    parser.add_argument("--require-refit", action="store_true")
    args = parser.parse_args()
    base, output = root / args.bundle, root / args.runs
    manifest, parent, original, catalog = load_bundle(root, base)
    config, followup = read_json(root / args.config), read_json(root / args.followup_config)
    for declaration in [config, followup]:
        validate_config(declaration, original["descriptor"]["protocol"])
        if declaration["feature_bundle_id"] != manifest["bundle_id"]:
            raise ValueError("The declared feature bundle differs.")
    if followup["adaptive_followup"]["parent_configuration_sha256"] != object_hash(config):
        raise ValueError("Follow-up declaration is not bound to the initial configuration.")
    code = model_code_receipt(root)
    references = reference_runs(root, root / args.reference_runs, config, original)
    groups = {"initial": (output, config), "followup": (root / args.followup_runs, followup)}
    experiments = {}
    for group, (directory, declaration) in groups.items():
        current = checked_runs(root, directory, declaration, manifest)
        names = declaration["variant_order"] + ([f"seed_{s}" for s in config["seed_replicas"]] if group == "initial" else [])
        if set(current) != {(name, fold) for name in names for fold in config["development_folds"]}:
            raise ValueError("Missing or undeclared development experiments.")
        experiments.update({(group, name, fold): item for (name, fold), item in current.items()})

    plan, selection = read_json(output / "seed_plan.json"), read_json(output / "selection.json")
    if (selection["config_sha256"] != object_hash(config) or selection["configuration"] != config
            or selection["followup_configuration"] != followup
            or selection["followup_configuration_sha256"] != object_hash(followup)
            or selection["model_code_sha256"] != code["sha256"]
            or selection["bundle_id"] != manifest["bundle_id"] or selection["audit_used"]
            or selection["seed_plan_sha256"] != sha256_file(output / "seed_plan.json")):
        raise ValueError("Selection provenance differs from the executed batches.")
    if (plan["config_sha256"] != object_hash(config) or plan["model_code_sha256"] != code["sha256"]
            or plan["bundle_id"] != manifest["bundle_id"] or plan["audit_used"]
            or plan["seed_replicas"] != config["seed_replicas"]):
        raise ValueError("Seed plan provenance changed.")

    metadata = []
    for source, count in manifest["rows"].items():
        data = pd.read_parquet(base / "features" / source, columns=[ID, "airport", "lirf_missing_record"])
        if len(data) != count or identity_hash(data[ID]) != manifest["source_ids_sha256"][source]:
            raise ValueError("Feature population differs from its frozen receipt.")
        data["source_file"] = source
        metadata.append(data)
    metadata = pd.concat(metadata, ignore_index=True)
    if metadata[ID].duplicated().any():
        raise ValueError("Feature populations overlap.")
    assignments, truths = {}, {}
    for fold in config["development_folds"]:
        assignments[fold] = pd.read_parquet(parent / "splits" / f"{fold}.parquet")
        _, truths[fold] = original_partition(parent, catalog, assignments[fold], "score")
    months = catalog[TIME].dt.strftime("%Y-%m")
    assignment = catalog[[ID]].copy()
    assignment["partition"] = np.where(months.isin(config["refit_months"]), "fit", "reserved")
    protected = set(catalog.loc[months.isin(original["descriptor"]["protocol"]["reserved_audit_months"]), "FLIGHT_ID_mvt"].dropna())
    assignment.loc[catalog["FLIGHT_ID_mvt"].isin(protected), "partition"] = "reserved"
    assignments["ranking"] = assignment
    populations, samples = {}, {}

    def population(fold, role, route):
        key = (fold, role, route)
        if key not in populations:
            ids = assignments[fold].loc[assignments[fold]["partition"].eq(role), ID]
            subset = metadata.loc[metadata[ID].isin(ids)]
            subset = subset.loc[route_mask(subset, route)]
            populations[key] = (len(subset), identity_hash(subset[ID]))
        return populations[key]

    def sampled_features(fold, predictions):
        if fold not in samples:
            candidates = metadata.loc[metadata[ID].isin(predictions[ID])]
            gated = candidates.loc[candidates["lirf_missing_record"].eq(1)]
            other = candidates.loc[~candidates[ID].isin(gated[ID])]
            chosen = pd.concat([gated, other.sample(n=min(max(0, 1000-len(gated)), len(other)), random_state=20260913)])
            pieces = []
            for source, subset in chosen.groupby("source_file"):
                frame = pd.read_parquet(base / "features" / source, columns=[ID, *FEATURE_NAMES])
                pieces.append(frame.loc[frame[ID].isin(subset[ID])])
            samples[fold] = pd.concat(pieces, ignore_index=True)
        return samples[fold]

    predictions_by_variant, scores, verified = {}, {}, []
    all_runs = [(group, path, report, groups[group][1]) for (group, _, _), (path, report) in experiments.items()]
    refits = [(file.parent, read_json(file)) for file in output.glob("*/run.json") if read_json(file)["stage"] == "refit"]
    if len(refits) > 1 or (args.require_refit and len(refits) != 1):
        raise ValueError("Expected exactly one finalized refit.")
    all_runs += [("initial", path, report, config) for path, report in refits]
    for group, path, run, declaration in all_runs:
        if (run["model_code"] != code or run["audit_used"] or run["bundle_id"] != manifest["bundle_id"]
                or run["parent_bundle_id"] != original["bundle_id"]
                or run["config_sha256"] != object_hash(declaration)
                or read_json(path / "config.json") != declaration):
            raise ValueError("Run provenance mismatch.")
        verify_receipts(path, run["artifacts"])
        pipeline = read_json(path / "pipeline.json")
        node_map = dict(iter_nodes(pipeline))
        if run["component_iterations"] != {name: node["iterations"] for name, node in node_map.items()}:
            raise ValueError("Component iteration receipts differ.")
        if run["stage"] == "development":
            fold, variant = run["fold"], run["variant"]
            for role in ["fit", "early_stop", "score"]:
                if (run[role + "_ids_sha256"] != original["folds"][fold]["identity_sha256"][role]
                        or run[role + "_rows"] != int(assignments[fold]["partition"].eq(role).sum())):
                    raise ValueError("A protected split changed.")
            predictions = pd.read_parquet(path / "predictions.parquet")
            metric = evaluate(truths[fold], predictions, original["ranking_month_weights"] if fold == "seasonal" else None)
            same_metrics(metric, read_json(path / "metrics.json"))
            scores[(group, variant, fold)] = metric
            if group == "initial" or variant != "reference":
                predictions_by_variant[(variant, fold)] = predictions
                published = selection["initial_and_seed_runs"][variant]["folds"][fold]
                if (published["run_id"] != run["run_id"] or published["run_sha256"] != sha256_file(path / "run.json")
                        or published["run_group"] != group):
                    raise ValueError("Selected experiment lineage changed.")
                same_metrics(metric, published["metrics"])
            seed = run["seed_override"]
            source = references[fold][0] if seed is None else experiments[("initial", plan["anchor_variant"], fold)][0]
            source_report = read_json(source / "run.json")
            lineage = {"run_id": source_report["run_id"], "run_sha256": sha256_file(source / "run.json")}
            if run["source_run"] != lineage:
                raise ValueError("Development source lineage changed.")
            if seed is not None:
                if variant != f"seed_{seed}" or seed not in config["seed_replicas"] or plan["anchor_runs"][fold]["run_sha256"] != lineage["run_sha256"]:
                    raise ValueError("Unexpected seed replica.")
            elif run["recipe"] != declaration["variants"][variant]:
                raise ValueError("Executed model specification differs from the declaration.")
            source_pipeline = read_json(source / "pipeline.json")
            if (pipeline["default"] != source_pipeline["default"] or pipeline["patches"] != source_pipeline["patches"]
                    or set(pipeline["nodes"]) != set(source_pipeline["nodes"])):
                raise ValueError("A component replacement changed the routing/blend structure.")
            for name, node in node_map.items():
                before = source_pipeline["nodes"][name]
                if any(node[k] != before[k] for k in ["schema", "route", "offset"]):
                    raise ValueError("A component replacement changed the feature or offset policy.")
                spec = run["recipe"]
                replaced = seed is not None or (spec["replace"] == "global" and name == pipeline["default"]) or (spec["replace"] == "airports" and name != pipeline["default"])
                if replaced:
                    expected_settings = deepcopy(before["settings"]) if seed is not None else derive_settings(declaration, before, spec)
                    if seed is not None:
                        expected_settings.update(seed=seed, threads=config["threads"])
                    expected_kind = before.get("model_kind", "lightgbm") if seed is not None else spec["kind"]
                    if node["settings"] != expected_settings or node["model_kind"] != expected_kind:
                        raise ValueError("Component settings differ from the declared replacement/seed.")
                else:
                    if node["copied_from"] != lineage or node["settings"] != before["settings"]:
                        raise ValueError("Imported component lineage differs.")
                    for key in ["model", "encoder", "history", "importance"]:
                        if key in node and sha256_file(path / node[key]) != sha256_file(source / before[key]):
                            raise ValueError("An unchanged component was modified.")
            if variant == "reference":
                pd.testing.assert_frame_equal(predictions.sort_values(ID).reset_index(drop=True), pd.read_parquet(source / "predictions.parquet").sort_values(ID).reset_index(drop=True), check_exact=True)
        elif run["stage"] == "refit":
            fold, metric = "ranking", None
            predictions = pd.read_parquet(path / "ranking_predictions.parquet")
            if (identity_hash(predictions[ID]) != original["ranking_ids_sha256"] or len(predictions) != original["ranking_departures"]
                    or predictions[ID].duplicated().any() or not np.isfinite(predictions["prediction"]).all()
                    or run["refit_months"] != config["refit_months"]
                    or (run["fit_rows"], run["fit_ids_sha256"]) != population(fold, "fit", "all")
                    or read_json(path / "selection.json") != selection):
                raise ValueError("Refit population or selection receipt changed.")
            expected_nodes = dict(iter_nodes(selection["refit_pipeline"]))
            if set(node_map) != set(expected_nodes) or pipeline["kind"] != "mixture":
                raise ValueError("Refit structure differs from the frozen selection.")
            if {m["name"]: m["weight"] for m in pipeline["members"]} != selection["candidates"][selection["selected_candidate"]]["weights"]:
                raise ValueError("Refit mixture weights differ.")
            expected_members = {m["name"]: m for m in selection["refit_pipeline"]["members"]}
            if len(pipeline["members"]) != len(expected_members):
                raise ValueError("Refit mixture has duplicate or missing members.")
            for member in pipeline["members"]:
                before = expected_members[member["name"]]["pipeline"]
                current = member["pipeline"]
                if current["default"] != before["default"] or current["patches"] != before["patches"]:
                    raise ValueError("Refit airport routing or internal blend weights changed.")
            for name, node in node_map.items():
                before = expected_nodes[name]
                if any(node[k] != before[k] for k in ["settings", "schema", "route", "offset"]):
                    raise ValueError("Refit component recipe differs.")
                if (node.get("model_kind", "lightgbm") != before.get("model_kind", "lightgbm")
                        or node["iterations"] != selection["refit_component_iterations"][name]
                        or node["early_stop_rows"] != 0 or node["early_stop_ids_sha256"] is not None):
                    raise ValueError("Refit inspected stopping data or changed tree counts.")
            unique = {node["model"]: node for node in node_map.values()}
            if run["unique_components"] != len(unique) or run["iterations"] != sum(node["iterations"] for node in unique.values()):
                raise ValueError("Deduplicated component count differs.")
        else:
            raise ValueError("Unrecognized model stage.")
        for node in node_map.values():
            for role in ["fit", "early_stop"] if run["stage"] == "development" else ["fit"]:
                if (node[role + "_rows"], node[role + "_ids_sha256"]) != population(fold, role, node["route"]):
                    raise ValueError("A component's fitting/stopping IDs differ from its route.")
        features = sampled_features(fold, predictions)
        restored = pipeline_prediction(path, pipeline, features, threads=4)
        expected = features[[ID]].merge(predictions, on=ID, validate="one_to_one", how="left")["prediction"].to_numpy()
        difference = float(np.max(np.abs(restored-expected)))
        if not np.isfinite(difference) or difference > 1e-9:
            raise ValueError("Native model predictions do not reproduce on this machine.")
        verified.append({"run_id": run["run_id"], "run_group": group, "stage": run["stage"], "components": len(node_map),
                         "reload_sample_rows": len(features), "reload_difference_seconds": difference,
                         "development_rmse_seconds": None if metric is None else metric["overall"]["rmse_seconds"]})
        print({"verified_run": run["run_id"], "reload_difference_seconds": difference}, flush=True)

    initial_ratios = {v: [scores[("initial", v, f)]["overall"]["rmse_seconds"] / scores[("initial", "reference", f)]["overall"]["rmse_seconds"] for f in config["development_folds"]] for v in config["variant_order"]}
    for variant, ratios in initial_ratios.items():
        published = selection["initial_and_seed_runs"][variant]
        if (not np.isclose(published["mean_relative_rmse"], np.mean(ratios), rtol=1e-12)
                or published["improves_both"] != all(r < 1 for r in ratios)):
            raise ValueError("Initial model ordering differs from its recomputed scores.")
    eligible_anchors = [v for v, ratios in initial_ratios.items() if v == "reference" or all(r < 1 for r in ratios)]
    anchor = min(eligible_anchors, key=lambda v: float(np.mean(initial_ratios[v])))
    if plan["anchor_variant"] != anchor:
        raise ValueError("Seed anchor does not follow the declared initial-only rule.")
    recipes = declared_weights(config, selection["initial_and_seed_runs"], anchor, followup)
    if set(recipes) != set(selection["candidates"]) or selection["candidate_count"] != len(recipes):
        raise ValueError("The saved blend search differs from its declared grid.")
    candidate_scores, blend_checks = {}, []
    for name, report in selection["candidates"].items():
        weights = report["weights"]
        validate_weights(list(weights.values()))
        if weights != recipes[name]:
            raise ValueError("Saved weights differ from the declared recipe.")
        ratios = []
        for fold in config["development_folds"]:
            detail = report["folds"][fold]
            artifact = detail["prediction_artifact"]
            verify_receipts(output, [artifact])
            saved = pd.read_parquet(output / artifact["path"])
            # Separate numerical implementation from the production blending function.
            parts = [predictions_by_variant[(v, fold)].set_index(ID).reindex(saved[ID])["prediction"].to_numpy() for v in weights]
            expected = np.column_stack(parts) @ np.asarray(list(weights.values()), dtype=float)
            difference = float(np.max(np.abs(expected-saved["prediction"].to_numpy())))
            if not np.isfinite(difference) or difference > 1e-9:
                raise ValueError("Saved blend differs from its ID-aligned components.")
            metrics = evaluate(truths[fold], saved, original["ranking_month_weights"] if fold == "seasonal" else None)
            same_metrics(metrics, detail["metrics"])
            ratio = metrics["overall"]["rmse_seconds"] / scores[("initial", "reference", fold)]["overall"]["rmse_seconds"]
            if not np.isclose(ratio, detail["rmse_ratio_to_v3"], rtol=1e-12, atol=0):
                raise ValueError("Reported improvement differs from the complete scores.")
            ratios.append(ratio)
            blend_checks.append({"candidate": name, "fold": fold, "rows": len(saved), "rmse_seconds": metrics["overall"]["rmse_seconds"], "maximum_prediction_difference_seconds": difference})
        eligible = all(r <= 1-config["minimum_improvement_fraction_each_fold"] for r in ratios)
        if eligible != report["eligible"] or not np.isclose(np.mean(ratios), report["mean_relative_rmse"], rtol=1e-12):
            raise ValueError("Promotion eligibility differs from the declared margin.")
        if eligible:
            candidate_scores[name] = float(np.mean(ratios))
    chosen = min(candidate_scores, key=candidate_scores.get) if candidate_scores else None
    if chosen != selection["selected_candidate"] or (not chosen and refits):
        raise ValueError("Selected model does not follow the promotion rule.")
    if chosen:
        frozen = dict(iter_nodes(selection["refit_pipeline"]))
        for qualified, node in frozen.items():
            variant, component = qualified.split("/", 1)
            group = "followup" if variant in followup["variants"] and variant != "reference" else "initial"
            sources = [read_json(experiments[(group, variant, fold)][0] / "pipeline.json")["nodes"][component] for fold in config["development_folds"]]
            count = int(np.rint(np.median([n["iterations"] for n in sources])))
            if selection["refit_component_iterations"][qualified] != count:
                raise ValueError("Refit count does not follow the median October rule.")
            for before in sources:
                if any(node[k] != before[k] for k in ["settings", "schema", "route", "offset"]):
                    raise ValueError("Refit blueprint differs from its two source folds.")

    previous = read_json(root / manifest["previous_bundle"] / "manifest.json")
    report = {"checked_at_utc": utc_now(), "verification_code_sha256": sha256_file(Path(__file__)),
              "model_code_sha256": code["sha256"], "bundle_id": manifest["bundle_id"], "audit_used": False,
              "parent_scientific_sha256": object_hash(scientific_parent(original)),
              "parent_artifacts_verified": len(original["artifacts"]), "previous_artifacts_verified": len(previous["artifacts"]),
              "feature_artifacts_verified": len(manifest["artifacts"]), "frozen_feature_rows_verified_by_hash_and_ids": len(metadata),
              "configuration_sha256": object_hash(config), "followup_configuration_sha256": object_hash(followup),
              "seed_plan_sha256": sha256_file(output / "seed_plan.json"), "selection_sha256": sha256_file(output / "selection.json"),
              "selected_candidate": chosen, "runs_verified": verified,
              "development_runs_rescored": sum(r["stage"] == "development" for r in verified), "blend_evaluations_verified": blend_checks,
              "reload_samples_include": "all gated LIRF scoring/ranking rows plus deterministic random rows, 1000 per fold",
              "maximum_native_reload_difference_seconds": max(r["reload_difference_seconds"] for r in verified),
              "maximum_blend_difference_seconds": max(r["maximum_prediction_difference_seconds"] for r in blend_checks)}
    write_json(root / args.output, report)
    print({"verified_runs": len(verified), "blend_evaluations": len(blend_checks), "selected_candidate": chosen,
           "maximum_native_reload_difference_seconds": report["maximum_native_reload_difference_seconds"],
           "maximum_blend_difference_seconds": report["maximum_blend_difference_seconds"]})


if __name__ == "__main__":
    main()
