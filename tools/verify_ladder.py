"""Independently verify ladder populations, complete scores, routes and native reloads."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.phase1 import validate_inputs
from prc2026.validation import evaluate, identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_nmclock.bundle import load_bundle
from prc2026_nmclock.features import FEATURE_NAMES
from prc2026_nmclock.inference import node_prediction as parent_node, pipeline_prediction as parent_prediction
from prc2026_ladder.inference import node_prediction, pipeline_prediction, recipe_pipeline, route_mask
from prc2026_ladder.training import (
    REFERENCE_NAMES, aligned_predictions, checked_runs, model_code_receipt, native_spec, reference_selection, validate_config,
)
from select_ladder import active_variants, collect, eligible_recipe
from refit_ladder import refit_assignment
from verify_phase5 import same_metrics


def difference(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Native comparisons require matching finite predictions.")
    delta = float(np.max(np.abs(left-right))) if left.size else 0.
    if delta > 1e-8:
        raise ValueError("Native prediction or recipe reconstruction differs.")
    return delta


def sample(frame, route):
    routed = frame.loc[route_mask(frame, route)].reset_index(drop=True)
    if route == "clock_missing":
        return routed
    exceptional = routed["nmclock_proxy_s"].lt(0) | routed["nmclock_proxy_s"].gt(10800)
    indexes = np.unique(np.r_[np.linspace(0, len(routed)-1, min(1000, len(routed)), dtype=int),
                             np.flatnonzero(exceptional), routed.groupby("airport", sort=True).head(1).index])
    return routed.iloc[indexes].reset_index(drop=True)


def check_card(node, variant, fitting, stopping=None, iterations=None):
    if node["spec"] != native_spec(variant) or node["settings"] != variant["settings"] or node["schema"] != variant["schema"]:
        raise ValueError("Native feature, route or model settings changed.")
    if node["fit_rows"] != len(fitting) or node["fit_ids_sha256"] != identity_hash(fitting[ID]):
        raise ValueError("Native fitting population changed.")
    if node["target_filter"] != "none":
        raise ValueError("Native fitting labels were filtered.")
    if stopping is None:
        if node["early_stop_rows"] or node["early_stop_ids_sha256"] is not None or node["iterations"] != iterations:
            raise ValueError("The refit used stopping rows or incorrect fixed iterations.")
    elif (node["early_stop_rows"] != len(stopping) or node["early_stop_ids_sha256"] != identity_hash(stopping[ID])
          or not 1 <= node["iterations"] <= variant["settings"][variant["kind"]]["rounds"]):
        raise ValueError("Native stopping population or selected iterations changed.")


def compare_selection(saved, rebuilt):
    for key in ["config_sha256", "model_code_sha256", "selection_code_sha256", "paired_day_code_sha256",
                "bundle_id", "parent_bundle_id", "candidate_count", "selected_candidate", "refit_iterations"]:
        if saved[key] != rebuilt[key]:
            raise ValueError("The complete selection no longer reproduces: "+key)
    same_metrics(rebuilt["candidates"], saved["candidates"])
    same_metrics(rebuilt["reference_metrics"], saved["reference_metrics"])
    same_metrics(rebuilt["selected_paired_days"], saved["selected_paired_days"])


def check_imported_sources(run, source, sources):
    source_nodes = read_json(source/"nodes.json")["cat_residual"]
    original = {"base": read_json(source/"pipeline.json"), "global": source_nodes["global"],
                "airports": {"kind": "nmclock_airports", "nodes": source_nodes["airports"]}}
    rewritten = deepcopy(original)
    checked = {}
    file_keys = {"model", "encoder", "history", "importance", "model_path", "encoder_path", "history_path", "importance_path"}
    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key in file_keys and isinstance(child, str):
                    if Path(child).name != child:
                        raise ValueError("Original native references must be flat files.")
                    name = "v6_"+child
                    if child not in checked:
                        actual, expected = sha256_file(run/name), sha256_file(source/child)
                        if actual != expected:
                            raise ValueError("An imported v6 native artifact changed.")
                        checked[child] = expected
                    value[key] = name
                else:
                    visit(child)
    visit(rewritten)
    expected_sources = {
        "v6": {"kind": "frozen_nmclock", "pipeline": rewritten["base"]},
        "v6_global": {"kind": "node", "node": rewritten["global"]},
        "v6_airports": {"kind": "frozen_nmclock", "pipeline": rewritten["airports"]},
    }
    for name, expected in expected_sources.items():
        if sources[name] != expected:
            raise ValueError("Imported v6 pipeline metadata or component weights changed.")
    return checked


def independently_reconstruct(run, source, features, candidate, new_nodes, threads):
    old_pipeline = read_json(source/"pipeline.json")
    old_nodes = read_json(source/"nodes.json")["cat_residual"]
    baseline = parent_prediction(source, old_pipeline, features, threads)
    new_values = np.empty(len(features))
    for route, weights in candidate["routes"].items():
        mask = route_mask(features, route)
        if not mask.any():
            continue
        query = features.loc[mask].reset_index(drop=True)
        new_values[mask] = 0.
        for name, weight in weights.items():
            if name == "v6":
                prediction = baseline[mask]
            elif name == "v6_global":
                prediction = parent_node(source, old_nodes["global"], query, threads)
            elif name == "v6_airports":
                prediction = parent_prediction(source, {"kind": "nmclock_airports", "nodes": old_nodes["airports"]}, query, threads)
            else:
                prediction = node_prediction(run, new_nodes[name], query, threads)
            new_values[mask] += weight*prediction
    weight = candidate["new_weight"]
    expected = new_values if weight == 1 else (1-weight)*baseline+weight*new_values
    for route, weights in candidate["routes"].items():
        if weights == {"v6": 1.0}:
            mask = route_mask(features, route)
            expected[mask] = baseline[mask]
    return expected


def verify(root, base, config_path, output, previous_runs, require_refit=False):
    config = read_json(config_path)
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(config, original["descriptor"]["protocol"])
    if manifest["bundle_id"] != config["feature_bundle_id"] or validate_inputs(root) != manifest["descriptor"]["input_files"]:
        raise ValueError("Original input files or inherited features changed.")
    saved = read_json(output/"selection.json")
    rebuilt = collect(root, base, config_path, output)
    compare_selection(saved, rebuilt)
    experiments = checked_runs(root, output, config, manifest)
    prior = reference_selection(root, config)
    native, scores, reference_checks, run_hashes = {}, {}, {}, {}
    for fold in config["development_folds"]:
        assignment = pd.read_parquet(parent/"splits"/(fold+".parquet"))
        features, truth = load_partition(base, parent, catalog, assignment, "score", FEATURE_NAMES)
        references = output/"references"/fold
        reference = aligned_predictions(features, pd.read_parquet(references/"v6.parquet"))
        original_card = prior["development_runs"][config["reference_variant"]][fold]
        source = previous_runs/original_card["run_id"]
        if sha256_file(source/"run.json") != original_card["run_sha256"]:
            raise ValueError("Original v6 native development source changed.")
        verify_receipts(source, read_json(source/"run.json")["artifacts"])
        original_prediction = aligned_predictions(features, pd.read_parquet(source/"predictions.parquet"))
        pd.testing.assert_frame_equal(original_prediction, reference, check_exact=True)
        query = sample(features, "clock_available")
        nodes = read_json(source/"nodes.json")
        deltas = {}
        for label, values in [
            ("v6_global", parent_node(source, nodes["global"], query, config["threads"])),
            ("v6_airports", parent_prediction(source, {"kind": "nmclock_airports", "nodes": nodes["airports"]}, query, config["threads"])),
        ]:
            recorded = query[[ID]].merge(pd.read_parquet(references/(label+".parquet")), on=ID, validate="one_to_one")["prediction"]
            deltas[label] = difference(values, recorded)
        reference_checks[fold] = {"sample_rows": len(query), "native_differences_seconds": deltas,
                                  "full_reference_rows": len(reference), "source_run_sha256": original_card["run_sha256"]}
        roles = {}
        for role in ["fit", "early_stop"]:
            roles[role], _ = load_partition(base, parent, catalog, assignment, role, ["nmclock_available", "nmclock_proxy_s"])
        for name in config["variant_order"]:
            directory, report = experiments[(name, fold)]
            variant = config["variants"][name]
            node = read_json(directory/"node.json")
            fitting = roles["fit"].loc[route_mask(roles["fit"], variant["route"])].reset_index(drop=True)
            stopping = roles["early_stop"].loc[route_mask(roles["early_stop"], variant["route"])].reset_index(drop=True)
            check_card(node, variant, fitting, stopping)
            for role, frame in [("fit", fitting), ("early_stop", stopping)]:
                if report["routed_"+role+"_ids_sha256"] != identity_hash(frame[ID]) or report["routed_"+role+"_rows"] != len(frame):
                    raise ValueError("Saved routed population receipt changed.")
            prediction = aligned_predictions(features, pd.read_parquet(directory/"predictions.parquet"))
            outside = ~route_mask(features, variant["route"])
            np.testing.assert_array_equal(prediction.loc[outside, "prediction"], reference.loc[outside, "prediction"])
            measured = evaluate(truth, prediction, original["ranking_month_weights"] if fold == "seasonal" else None)
            same_metrics(measured, read_json(directory/"metrics.json"))
            query = sample(features, variant["route"])
            values = node_prediction(directory, node, query, config["threads"])
            recorded = query[[ID]].merge(prediction, on=ID, validate="one_to_one")["prediction"]
            key = name+"/"+fold
            native[key] = {"rows": len(query), "max_difference_seconds": difference(values, recorded)}
            scores[key] = measured["overall"]["rmse_seconds"]
            run_hashes[key] = sha256_file(directory/"run.json")
    refits = [(path.parent, read_json(path)) for path in output.glob("*/run.json") if read_json(path)["stage"] == "refit"]
    if len(refits) > 1 or (require_refit and not refits):
        raise ValueError("Exactly one completed refit is required when requested.")
    refit_report = None
    if refits:
        run, report = refits[0]
        code = model_code_receipt(root)
        candidate = eligible_recipe(config, saved, manifest, code)
        verify_receipts(run, report["artifacts"])
        if (report["model_code"]["sha256"] != code["sha256"] or report["selection_sha256"] != sha256_file(output/"selection.json")
                or report["config_sha256"] != object_hash(config) or report["audit_used"]
                or report["refit_months"] != config["refit_months"] or report["refit_code_sha256"] != sha256_file(root/"tools/refit_ladder.py")):
            raise ValueError("Refit provenance changed.")
        source = previous_runs/config["reference_refit_run"]
        if sha256_file(source/"run.json") != config["reference_refit_sha256"]:
            raise ValueError("Frozen v6 refit changed.")
        verify_receipts(source, read_json(source/"run.json")["artifacts"])
        nodes, sources = read_json(run/"nodes.json"), read_json(run/"sources.json")
        names = active_variants(config, candidate)
        if set(nodes) != set(names) or set(sources) != REFERENCE_NAMES | set(names):
            raise ValueError("Refit native component inventory differs from the selected family.")
        imported = check_imported_sources(run, source, sources)
        expected_pipeline = recipe_pipeline(candidate["routes"], candidate["new_weight"], sources)
        if expected_pipeline != read_json(run/"pipeline.json"):
            raise ValueError("Refit weights or routes differ from the selected recipe.")
        assignment = refit_assignment(catalog, original["descriptor"]["protocol"], config["refit_months"])
        fitting, _ = load_partition(base, parent, catalog, assignment, "fit", ["nmclock_available", "nmclock_proxy_s"])
        if len(fitting) != report["fit_rows"] or identity_hash(fitting[ID]) != report["fit_ids_sha256"]:
            raise ValueError("Protected refit population changed.")
        for name in names:
            variant = config["variants"][name]
            frame = fitting.loc[route_mask(fitting, variant["route"])].reset_index(drop=True)
            check_card(nodes[name], variant, frame, iterations=saved["refit_iterations"][name])
            if sources[name] != {"kind": "node", "node": nodes[name]}:
                raise ValueError("A new native refit source was substituted.")
        ranking = pd.read_parquet(base/"features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
        prediction = aligned_predictions(ranking, pd.read_parquet(run/"ranking_predictions.parquet"))
        indexes = np.unique(np.r_[np.linspace(0, len(ranking)-1, 1000, dtype=int),
                                 np.flatnonzero(route_mask(ranking, "clock_missing")),
                                 np.flatnonzero(ranking["lirf_missing_record"].eq(1)),
                                 np.flatnonzero(ranking["nmclock_proxy_s"].gt(10800) | ranking["nmclock_proxy_s"].lt(0))])
        query = ranking.iloc[indexes].reset_index(drop=True)
        values = pipeline_prediction(run, read_json(run/"pipeline.json"), query, config["threads"])
        expected = independently_reconstruct(run, source, query, candidate, nodes, config["threads"])
        refit_report = {"run_id": run.name, "run_sha256": sha256_file(run/"run.json"), "rows": len(query),
                        "ranking_rows": len(ranking), "missing_clock_sample_rows": int(route_mask(query, "clock_missing").sum()),
                        "max_prediction_difference_seconds": difference(values, prediction.iloc[indexes]["prediction"]),
                        "max_recipe_reconstruction_difference_seconds": difference(values, expected),
                        "imported_native_files_unchanged": len(imported)}
    return {"created_at_utc": utc_now(), "passed": True, "audit_used": False,
            "bundle_id": manifest["bundle_id"], "model_code_sha256": model_code_receipt(root)["sha256"],
            "verification_code_sha256": sha256_file(Path(__file__)),
            "selection_sha256": sha256_file(output/"selection.json"), "selected_candidate": saved["selected_candidate"],
            "inherited_feature_rows_hash_verified": sum(manifest["rows"].values()),
            "development_run_count": len(experiments), "candidate_count": saved["candidate_count"],
            "complete_scores": scores, "reference_checks": reference_checks,
            "native_prediction_checks": native, "run_sha256": run_hashes, "refit": refit_report}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("data/nmclock/72e86725e23367ede5b5"))
    parser.add_argument("--config", type=Path, default=Path("configs/ladder_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/ladder"))
    parser.add_argument("--previous-runs", type=Path, default=Path("runs/nmclock"))
    parser.add_argument("--output", type=Path, default=Path("docs/ladder/verification.json"))
    parser.add_argument("--require-refit", action="store_true")
    args = parser.parse_args()
    report = verify(root, root/args.bundle, root/args.config, root/args.runs, root/args.previous_runs, args.require_refit)
    path = root/args.output
    if path.exists():
        raise ValueError("Preserve the original verification report; choose a new output path.")
    write_json(path, report)
    print({"passed": report["passed"], "development_runs": report["development_run_count"],
           "candidate_count": report["candidate_count"], "selected": report["selected_candidate"],
           "refit": report["refit"]})


if __name__ == "__main__":
    main()
