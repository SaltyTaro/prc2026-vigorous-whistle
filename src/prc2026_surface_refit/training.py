"""Fit the twelve selected components once, at their frozen October counts."""
import gc
from pathlib import Path
import shutil
import socket

import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID
from prc2026.phase1 import emit
from prc2026.validation import identity_hash
from prc2026_models.experiments import model_environment, new_run, receipts
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_nmclock.features import route_mask
from prc2026_nmclock.inference import copy_file_tree
from prc2026_timeline.bundle import load_bundle
from prc2026_timeline.features import FEATURE_NAMES
from prc2026_timeline.native import fit_node
from prc2026_timeline.training import checked_runs as timeline_runs_checked
from prc2026_surface_airports.training import checked_runs, model_code_receipt as development_code, validate_config
from .guards import eligible_recipe, refit_assignment, require_verifications
from .inference import SELECTED, make_pipeline, pipeline_prediction


def model_code_receipt(root):
    files = dict(development_code(root)["files"])
    paths = list((root / "src/prc2026_surface_refit").glob("*.py")) + [root / "tools/refit_surface.py"]
    files.update({path.relative_to(root).as_posix(): sha256_file(path) for path in paths})
    return {"files": files, "sha256": object_hash(files)}


def preflight(root, refit_config_path, development_runs, timeline_runs, previous_runs):
    refit_config = read_json(refit_config_path)
    for field in ["selection", "development_config", "timeline_config"]:
        if sha256_file(root / refit_config[field]) != refit_config[field + "_sha256"]:
            raise ValueError("An input declaration or selection changed: " + field)
    if (refit_config["audit_used"] or refit_config["automatic_submission"]
            or refit_config["worker"] != "arcw0" or refit_config["cloud"] != "Azure"
            or refit_config["selected_candidate"] != SELECTED or refit_config["new_native_models"] != 12):
        raise ValueError("The bounded selected refit declaration changed.")
    config, selection = [read_json(root / refit_config[key]) for key in ["development_config", "selection"]]
    base = root / refit_config["feature_bundle"]
    manifest, parent, original, catalog = load_bundle(root, base)
    validate_config(root, config, original["descriptor"]["protocol"])
    if sha256_file(base / "manifest.json") != config["feature_manifest_sha256"]:
        raise ValueError("The frozen feature manifest changed.")
    timeline_config = read_json(root / refit_config["timeline_config"])
    timeline = read_json(root / config["timeline_selection"])
    code = development_code(root)
    eligible_recipe(config, selection, timeline, manifest, code)
    if selection["selection_code_sha256"] != sha256_file(root / "tools/select_surface_airports.py"):
        raise ValueError("Selection implementation changed.")
    verification_receipts = require_verifications(root, root / refit_config["selection"], selection, code)
    if selection["refit_iterations"] != refit_config["component_iterations"]:
        raise ValueError("The declared fixed refit counts changed.")
    experiments = checked_runs(root, development_runs, config, manifest)
    if set(experiments) != set(config["development_folds"]):
        raise ValueError("Both complete airport folds must precede refitting.")
    for fold, (directory, _) in experiments.items():
        saved = selection["development_runs"][fold]
        if directory.name != saved["run_id"] or sha256_file(directory / "run.json") != saved["run_sha256"]:
            raise ValueError("An airport development run changed.")
    globals_settings = {key: timeline_config[key] for key in ["threads", "seed", "lightgbm", "catboost"]}
    globals_experiments = timeline_runs_checked(root, timeline_runs, timeline_config, manifest)
    if set(globals_experiments) != {(name, fold) for name in timeline_config["variant_order"] for fold in config["development_folds"]}:
        raise ValueError("The complete global feature comparison is required.")
    for name in ["cat_combined", "lgb_combined"]:
        for fold in config["development_folds"]:
            directory, report = globals_experiments[(name, fold)]
            saved = timeline["development_runs"][name][fold]
            node = read_json(directory / "node.json")
            if (directory.name != saved["run_id"] or sha256_file(directory / "run.json") != saved["run_sha256"]
                    or node["spec"] != timeline_config["variants"][name] or node["settings"] != globals_settings):
                raise ValueError("A selected global learner or its settings changed.")
    source = previous_runs / refit_config["reference_refit_run"]
    if sha256_file(source / "run.json") != refit_config["reference_refit_sha256"]:
        raise ValueError("The immutable v6 refit changed.")
    source_run = read_json(source / "run.json")
    verify_receipts(source, source_run["artifacts"])
    return {"refit_config": refit_config, "config": config, "selection": selection, "base": base,
            "manifest": manifest, "parent": parent, "original": original, "catalog": catalog,
            "timeline_config": timeline_config, "globals_settings": globals_settings,
            "source": source, "source_run": source_run, "verification_receipts": verification_receipts}


def refit(root, refit_config_path, output, development_runs, timeline_runs, previous_runs):
    if socket.gethostname() != "arcw0" or root.resolve() != Path("/home/azureuser/prc-flight-2026"):
        raise ValueError("Training is restricted to the existing isolated arcw0 workspace.")
    if list(output.glob("*_refit_*")):
        raise ValueError("Preserve every completed or partial refit; do not duplicate it.")
    context = preflight(root, refit_config_path, development_runs, timeline_runs, previous_runs)
    config, selection, refit_config = [context[key] for key in ["config", "selection", "refit_config"]]
    base, parent, original, catalog = [context[key] for key in ["base", "parent", "original", "catalog"]]
    if shutil.disk_usage(root).free < 900_000_000:
        raise ValueError("Less than 900 MB free before the selected refit.")
    run = new_run(output, "refit_" + SELECTED)
    with ResourceMonitor() as monitor:
        assignment = refit_assignment(catalog, original["descriptor"]["protocol"], config["refit_months"])
        features, truth = load_partition(base, parent, catalog, assignment, "fit", FEATURE_NAMES)
        truth = aligned_truth(features, truth)
        mask = route_mask(features)
        routed, routed_truth = features.loc[mask].reset_index(drop=True), truth.loc[mask].reset_index(drop=True)
        for name, frame in [("fit", features), ("clock_fit", routed)]:
            if len(frame) != context["source_run"][name + "_rows"] or identity_hash(frame[ID]) != context["source_run"][name + "_ids_sha256"]:
                raise ValueError("The refit population differs from the December-protected v6 population.")
        assignment.to_parquet(run / "refit_assignment.parquet", index=False, compression="zstd")
        for name, payload in [("selection", selection), ("config", config), ("refit_config", refit_config)]:
            write_json(run / (name + ".json"), payload)
        reference = copy_file_tree(run, context["source"], read_json(context["source"] / "pipeline.json"), "v6_")
        nodes = {"globals": {}, "airports": {}}
        for name in ["cat_combined", "lgb_combined"]:
            nodes["globals"][name] = fit_node(run, name, context["timeline_config"]["variants"][name],
                context["globals_settings"], routed, routed_truth, iterations=selection["refit_iterations"]["globals"][name])
            write_json(run / "nodes_progress.json", nodes)
            emit("surface_refit_global_complete", component=name, iterations=nodes["globals"][name]["iterations"])
            gc.collect()
        for airport in config["airports"]:
            if shutil.disk_usage(root).free < 750_000_000:
                raise ValueError("Less than 750 MB free before another selected airport fit.")
            airport_mask = routed.airport.eq(airport)
            nodes["airports"][airport] = fit_node(run, "airport_" + airport, config["native_spec"], config["settings"],
                routed.loc[airport_mask].reset_index(drop=True), routed_truth.loc[airport_mask].reset_index(drop=True),
                iterations=selection["refit_iterations"]["airports"][airport])
            write_json(run / "nodes_progress.json", nodes)
            emit("surface_refit_airport_complete", airport=airport, iterations=nodes["airports"][airport]["iterations"])
            gc.collect()
        pipeline = make_pipeline(nodes, reference)
        write_json(run / "nodes.json", nodes)
        write_json(run / "pipeline.json", pipeline)
        ranking = pd.read_parquet(base / "features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking movement identities changed.")
        values = pipeline_prediction(run, pipeline, ranking, config["threads"])
        pd.DataFrame({ID: ranking[ID], "prediction": values}).to_parquet(run / "ranking_predictions.parquet", index=False, compression="zstd")
        unusual = (ranking.nmclock_proxy_s.lt(0) | ranking.nmclock_proxy_s.gt(10800)
                   | ranking.nmclock_clock_difference_s.abs().gt(3600) | ranking.lirf_missing_record.eq(1))
        missing = ~route_mask(ranking)
        indexes = np.unique(np.r_[np.linspace(0, len(ranking) - 1, 1000, dtype=int), np.flatnonzero(missing | unusual)])
        restored = pipeline_prediction(run, read_json(run / "pipeline.json"), ranking.iloc[indexes].reset_index(drop=True), config["threads"])
        delta = float(np.max(np.abs(values[indexes] - restored)))
        if delta > 1e-8:
            raise ValueError("The saved selected pipeline does not reproduce.")
        reference_values = ranking[[ID]].merge(pd.read_parquet(context["source"] / "ranking_predictions.parquet"), on=ID, how="left", validate="one_to_one")["prediction"].to_numpy()
        if not np.array_equal(values[missing], reference_values[missing]):
            raise ValueError("Every missing-clock ranking prediction must remain bit-identical to v6.")
    report = {"created_at_utc": utc_now(), "run_id": run.name, "stage": "refit", "model_kind": "ensemble",
              "variant": SELECTED, "iterations": sum(sum(row.values()) for row in selection["refit_iterations"].values()),
              "iterations_scope": "Twelve newly fitted native components; copied v6 models retain their original counts.",
              "component_iterations": selection["refit_iterations"], "newly_fitted_models": 12,
              "bundle_id": context["manifest"]["bundle_id"], "parent_bundle_id": original["bundle_id"],
              "model_code": model_code_receipt(root), "development_model_code_sha256": development_code(root)["sha256"],
              "refit_config_sha256": object_hash(refit_config), "config_sha256": object_hash(config),
              "selection_sha256": sha256_file(root / refit_config["selection"]),
              "source_refit_sha256": refit_config["reference_refit_sha256"], "verification_receipts": context["verification_receipts"],
              "fit_rows": len(features), "fit_ids_sha256": identity_hash(features[ID]),
              "clock_fit_rows": len(routed), "clock_fit_ids_sha256": identity_hash(routed[ID]),
              "refit_months": config["refit_months"], "audit_used": False, "target_filter": "none", "prediction_postprocess": "none",
              "ranking_rows": len(ranking), "ranking_ids_sha256": identity_hash(ranking[ID]),
              "reload_sample_rows": len(indexes), "max_model_reload_difference_seconds": delta,
              "fallback_ranking_rows": int(missing.sum()), "max_fallback_difference_seconds": 0.,
              "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run)}
    write_json(run / "run.json", report)
    emit("surface_refit_complete", run=str(run), ranking_rows=len(ranking), resources=monitor.stats)
    return run
