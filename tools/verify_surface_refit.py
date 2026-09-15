"""Verify protected refitting and reconstruct every ranking prediction natively."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.phase1 import validate_inputs
from prc2026.validation import identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_nmclock.features import route_mask
from prc2026_timeline.features import FEATURE_NAMES, CATEGORICAL_COLUMNS
from prc2026_timeline.encoding import contract
from prc2026_timeline.inference import node_prediction
from prc2026_surface_refit.guards import refit_assignment
from prc2026_surface_refit.inference import SELECTED, make_pipeline, pipeline_prediction
from prc2026_surface_refit.training import model_code_receipt, preflight


def verify_copied_reference(run, source):
    reference = deepcopy(read_json(source / "pipeline.json"))
    file_keys = {"model", "encoder", "history", "importance", "model_path", "encoder_path", "history_path", "importance_path"}
    checked = {}
    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key in file_keys and isinstance(child, str):
                    if Path(child).name != child:
                        raise ValueError("A copied native path escapes its run.")
                    digest = sha256_file(source / child)
                    if sha256_file(run / ("v6_" + child)) != digest:
                        raise ValueError("An imported v6 model artifact changed.")
                    checked["v6_" + child] = digest
                    value[key] = "v6_" + child
                else:
                    visit(child)
    visit(reference)
    return reference, checked


def check_node(run, node, fitting, spec, settings, iterations):
    if (node["schema"] != "timeline" or node["spec"] != spec or node["settings"] != settings
            or node["target_filter"] != "none" or node["iterations"] != iterations
            or node["fit_rows"] != len(fitting) or node["fit_ids_sha256"] != identity_hash(fitting[ID])
            or node["early_stop_rows"] != 0 or node["early_stop_ids_sha256"] is not None):
        raise ValueError("A refitted native component changed its data, settings or fixed count.")
    encoder = read_json(run / node["encoder_path"])
    names, cats = contract(spec)
    vocab = {name: sorted(set(fitting[name].astype("string").fillna("__MISSING__")) | {"__MISSING__"}) for name in cats}
    if encoder["feature_names"] != names or encoder["categorical_columns"] != cats or encoder["vocabularies"] != vocab:
        raise ValueError("A refit encoder differs from the protected fitting-only vocabulary.")
    return {"fit_rows": len(fitting), "fit_ids_sha256": identity_hash(fitting[ID]),
            "fixed_iterations": iterations, "early_stop_rows": 0, "fit_encoder_verified": True}


def verify(runs, development_runs, timeline_runs, previous_runs):
    context = preflight(ROOT, ROOT / "configs/surface_refit_v1.json", development_runs, timeline_runs, previous_runs)
    candidates = [path.parent for path in runs.glob("*/run.json") if read_json(path)["stage"] == "refit"]
    if len(candidates) != 1 or len(list(runs.glob("*_refit_*"))) != 1:
        raise ValueError("Exactly one complete selected refit is required.")
    run = candidates[0]
    report = read_json(run / "run.json")
    verify_receipts(run, report["artifacts"])
    config, selection, base, parent, original, catalog = [context[key] for key in ["config", "selection", "base", "parent", "original", "catalog"]]
    if validate_inputs(ROOT) != context["manifest"]["descriptor"]["input_files"]:
        raise ValueError("Original competition inputs changed.")
    if (report["audit_used"] or report["variant"] != SELECTED or report["newly_fitted_models"] != 12
            or report["model_code"] != model_code_receipt(ROOT)
            or report["config_sha256"] != object_hash(config)
            or report["refit_config_sha256"] != object_hash(context["refit_config"])
            or report["source_refit_sha256"] != context["refit_config"]["reference_refit_sha256"]
            or report["selection_sha256"] != sha256_file(ROOT / context["refit_config"]["selection"])
            or report["verification_receipts"] != context["verification_receipts"]
            or report["bundle_id"] != context["manifest"]["bundle_id"]
            or report["refit_months"] != config["refit_months"]
            or report["component_iterations"] != selection["refit_iterations"]
            or report["target_filter"] != "none" or report["prediction_postprocess"] != "none"):
        raise ValueError("The final refit differs from the selected and verified recipe.")
    if read_json(run / "selection.json") != selection or read_json(run / "config.json") != config or read_json(run / "refit_config.json") != context["refit_config"]:
        raise ValueError("The saved refit declarations differ from their original evidence.")
    assignment = refit_assignment(catalog, original["descriptor"]["protocol"], config["refit_months"])
    saved_assignment = pd.read_parquet(run / "refit_assignment.parquet")
    if not assignment.equals(saved_assignment):
        raise ValueError("The saved refit assignment differs from the protected December flight groups.")
    columns = list(dict.fromkeys(["airport", "nmclock_available", "nmclock_proxy_s", *CATEGORICAL_COLUMNS]))
    fitting, _ = load_partition(base, parent, catalog, assignment, "fit", columns)
    routed = fitting.loc[route_mask(fitting)].reset_index(drop=True)
    for name, frame in [("fit", fitting), ("clock_fit", routed)]:
        if report[name + "_rows"] != len(frame) or report[name + "_ids_sha256"] != identity_hash(frame[ID]):
            raise ValueError("A complete fitting population changed.")
        if report[name + "_rows"] != context["source_run"][name + "_rows"] or report[name + "_ids_sha256"] != context["source_run"][name + "_ids_sha256"]:
            raise ValueError("Fitting differs from the original protected v6 population.")
    nodes = read_json(run / "nodes.json")
    if set(nodes) != {"globals", "airports"} or set(nodes["globals"]) != {"cat_combined", "lgb_combined"} or set(nodes["airports"]) != set(config["airports"]):
        raise ValueError("All twelve selected native components are required.")
    node_checks = {"globals": {}, "airports": {}}
    for name, node in nodes["globals"].items():
        node_checks["globals"][name] = check_node(run, node, routed, context["timeline_config"]["variants"][name],
                                                context["globals_settings"], selection["refit_iterations"]["globals"][name])
    for airport, node in nodes["airports"].items():
        airport_fit = routed.loc[routed.airport.eq(airport)].reset_index(drop=True)
        node_checks["airports"][airport] = check_node(run, node, airport_fit, config["native_spec"], config["settings"], selection["refit_iterations"]["airports"][airport])
    if report["iterations"] != sum(check["fixed_iterations"] for family in node_checks.values() for check in family.values()):
        raise ValueError("Total new native tree count changed.")
    reference, copied_files = verify_copied_reference(run, context["source"])
    pipeline = read_json(run / "pipeline.json")
    if pipeline != make_pipeline(nodes, reference):
        raise ValueError("The portable final pipeline differs from its native components and frozen fallback.")
    ranking = pd.read_parquet(base / "features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
    saved = pd.read_parquet(run / "ranking_predictions.parquet")
    if (list(saved.columns) != [ID, "prediction"] or not ranking[ID].equals(saved[ID])
            or report["ranking_rows"] != len(ranking) or len(ranking) != 344841
            or report["ranking_ids_sha256"] != identity_hash(ranking[ID])
            or report["ranking_ids_sha256"] != original["ranking_ids_sha256"]):
        raise ValueError("The complete ranking population or ordering changed.")
    restored = pipeline_prediction(run, pipeline, ranking, config["threads"])
    delta = float(np.max(np.abs(restored - saved.prediction.to_numpy())))
    if not np.isfinite(delta) or delta > 1e-8:
        raise ValueError("The portable native pipeline fails complete ranking reconstruction.")
    available = route_mask(ranking)
    routed_ranking = ranking.loc[available].reset_index(drop=True)
    globals_pred = {name: node_prediction(run, node, routed_ranking, config["threads"]) for name, node in nodes["globals"].items()}
    airports_pred = np.full(len(routed_ranking), np.nan)
    for airport, node in nodes["airports"].items():
        mask = routed_ranking.airport.eq(airport).to_numpy()
        airports_pred[mask] = node_prediction(run, node, routed_ranking.loc[mask].reset_index(drop=True), config["threads"])
        node_checks["airports"][airport]["ranking_rows_reconstructed"] = int(mask.sum())
    expected = ranking[[ID]].merge(pd.read_parquet(context["source"] / "ranking_predictions.parquet"), on=ID, how="left", validate="one_to_one")["prediction"].to_numpy()
    independent = expected.copy()
    independent[available] = .25 * globals_pred["cat_combined"] + .25 * globals_pred["lgb_combined"] + .5 * airports_pred
    weight_delta = float(np.max(np.abs(independent - saved.prediction.to_numpy())))
    if not np.isfinite(weight_delta) or weight_delta > 1e-8:
        raise ValueError("The independent 25/25/50 native reconstruction differs.")
    if not np.array_equal(saved.prediction.to_numpy()[~available], expected[~available]):
        raise ValueError("At least one missing-clock ranking prediction changed from v6.")
    for row in node_checks["globals"].values():
        row["ranking_rows_reconstructed"] = int(available.sum())
    return {"created_at_utc": utc_now(), "passed": True, "audit_used": False, "run_id": run.name,
            "run_sha256": sha256_file(run / "run.json"), "selection_sha256": report["selection_sha256"],
            "model_code_sha256": report["model_code"]["sha256"], "verifier_code_sha256": sha256_file(Path(__file__)),
            "bundle_id": report["bundle_id"], "selected_candidate": SELECTED, "newly_fitted_models": 12,
            "complete_ranking_rows": len(ranking), "ranking_ids_sha256": identity_hash(ranking[ID]),
            "complete_native_reload_maximum_difference_seconds": delta, "independent_weight_maximum_difference_seconds": weight_delta,
            "fallback_rows_identical_to_v6": int((~available).sum()), "copied_v6_files": copied_files,
            "fit_rows": len(fitting), "fit_ids_sha256": identity_hash(fitting[ID]),
            "clock_fit_rows": len(routed), "clock_fit_ids_sha256": identity_hash(routed[ID]),
            "refit_assignment_sha256": sha256_file(run / "refit_assignment.parquet"),
            "component_checks": node_checks, "development_verification_receipts": report["verification_receipts"],
            "interpretation": "No ranking targets were read. Complete native ranking reconstruction and fixed recipe verification; December remains reserved."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/surface_refit"))
    parser.add_argument("--development-runs", type=Path, default=Path("runs/surface_airports"))
    parser.add_argument("--timeline-runs", type=Path, default=Path("runs/timeline"))
    parser.add_argument("--previous-runs", type=Path, default=Path("runs/nmclock"))
    parser.add_argument("--output", type=Path, default=Path("docs/surface_refit/verification.json"))
    args = parser.parse_args()
    destination = ROOT / args.output
    if destination.exists():
        raise ValueError("Preserve the original final verification report.")
    report = verify(ROOT / args.runs, ROOT / args.development_runs, ROOT / args.timeline_runs, ROOT / args.previous_runs)
    write_json(destination, report)
    print({key: report[key] for key in ["passed", "run_id", "complete_ranking_rows", "complete_native_reload_maximum_difference_seconds", "independent_weight_maximum_difference_seconds", "fallback_rows_identical_to_v6"]})


if __name__ == "__main__":
    main()
