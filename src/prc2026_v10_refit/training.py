"""Fixed-iteration Jan-Nov refit of the ladder variant cat_long for the selected v10 candidate.

Runs only on the authorized arcw0 workspace. Fits one CatBoost on the same
December-protected clock-present population as the frozen v6 refit, at the
declared iteration count, predicts the clock-available ranking rows and checks
that the saved model reproduces its predictions after reload.
"""
from pathlib import Path
import shutil
import socket

import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID
from prc2026.phase1 import emit
from prc2026.validation import identity_hash
from prc2026_models.experiments import model_environment, new_run, receipts
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_nmclock.bundle import load_bundle
from prc2026_nmclock.features import FEATURE_NAMES
from prc2026_ladder.inference import node_prediction, route_mask
from prc2026_ladder.training import fit_variant, model_code_receipt as ladder_code
from prc2026_surface_refit.guards import refit_assignment

WORKSPACE = Path("/home/azureuser/prc-flight-2026")


def refit_code_receipt(root: Path) -> dict:
    files = dict(ladder_code(root)["files"])
    paths = sorted((root / "src/prc2026_v10_refit").glob("*.py")) + [root / "tools/refit_v10_cat_long.py"]
    files.update({path.relative_to(root).as_posix(): sha256_file(path) for path in paths})
    return {"files": files, "sha256": object_hash(files)}


def preflight(root: Path, config_path: Path, selection_path: Path) -> dict:
    config, selection = read_json(config_path), read_json(selection_path)
    if selection["config_sha256"] != object_hash(config) or selection["audit_used"]:
        raise ValueError("The selection does not belong to this declaration.")
    if not selection["selected_candidate"] or "cat_long_25" not in selection["selected_items"]:
        raise ValueError("The selected candidate does not need the cat_long refit.")
    spec = config["cat_long_refit"]
    ladder_config = read_json(root / spec["ladder_config"])
    if sha256_file(root / spec["ladder_config"]) != config["parents"]["ladder_config"]["sha256"]:
        raise ValueError("The ladder declaration changed.")
    variant = ladder_config["variants"][spec["variant"]]
    if variant["route"] != "clock_available" or variant["target"] != "nm_residual" or variant["schema"] != "nmclock" or variant["kind"] != "catboost":
        raise ValueError("Unexpected cat_long variant definition.")
    if variant["settings"]["threads"] != spec["threads"] or variant["settings"]["catboost"]["parameters"]["task_type"] != "CPU":
        raise ValueError("Declared CPU settings differ from the ladder variant.")
    if ladder_config["feature_bundle_id"] != spec["bundle_id"]:
        raise ValueError("The ladder bundle differs from the declared refit bundle.")
    iterations = []
    for fold in config["development_folds"]:
        run = read_json(root / config["parents"][f"ladder_{fold}_cat_long_run"]["path"])
        if run["variant"] != spec["variant"] or run["bundle_id"] != spec["bundle_id"] or run["audit_used"]:
            raise ValueError("A ladder development run does not match the declaration.")
        iterations.append(run["best_iterations"])
    if spec["iterations"] != int(np.rint(np.median(iterations))) or spec["iterations"] < 1:
        raise ValueError("Refit iterations must be the rounded median of the development counts.")
    base = root / spec["bundle"]
    manifest, parent, original, catalog = load_bundle(root, base)
    if manifest["bundle_id"] != spec["bundle_id"]:
        raise ValueError("The NM clock feature bundle identity changed.")
    source = root / spec["reference_refit_run"]
    source_run = read_json(source / "run.json")
    if source_run["stage"] != "refit" or source_run["clock_fit_rows"] != spec["expected_clock_fit_rows"] or source_run["clock_fit_ids_sha256"] != spec["expected_clock_fit_ids_sha256"]:
        raise ValueError("The frozen v6 refit population differs from the declaration.")
    return {"config": config, "selection": selection, "spec": spec, "variant": variant, "base": base, "manifest": manifest,
            "parent": parent, "original": original, "catalog": catalog, "source_run": source_run}


def refit(root: Path, config_path: Path, selection_path: Path, output: Path) -> Path:
    if socket.gethostname() != "arcw0" or root.resolve() != WORKSPACE:
        raise ValueError("Training is restricted to the existing isolated arcw0 workspace.")
    if list(output.glob("*_refit_*")):
        raise ValueError("Preserve every completed or partial refit; do not duplicate it.")
    context = preflight(root, config_path, selection_path)
    config, spec, variant = context["config"], context["spec"], context["variant"]
    base, parent, original, catalog = context["base"], context["parent"], context["original"], context["catalog"]
    if shutil.disk_usage(root).free < 400_000_000:
        raise ValueError("Less than 400 MB free before the refit.")
    run = new_run(output, "refit_" + spec["variant"])
    with ResourceMonitor() as monitor:
        assignment = refit_assignment(catalog, original["descriptor"]["protocol"], config["refit_months"])
        features, truth = load_partition(base, parent, catalog, assignment, "fit", FEATURE_NAMES)
        truth = aligned_truth(features, truth)
        mask = route_mask(features, "clock_available")
        routed, routed_truth = features.loc[mask].reset_index(drop=True), truth.loc[mask].reset_index(drop=True)
        if len(routed) != spec["expected_clock_fit_rows"] or identity_hash(routed[ID]) != spec["expected_clock_fit_ids_sha256"]:
            raise ValueError("The refit population differs from the December-protected v6 clock population.")
        if len(features) != context["source_run"]["fit_rows"] or identity_hash(features[ID]) != context["source_run"]["fit_ids_sha256"]:
            raise ValueError("The complete refit population differs from the frozen v6 refit.")
        del features, truth
        for name, payload in [("config", config), ("selection", context["selection"]), ("variant", variant)]:
            write_json(run / (name + ".json"), payload)
        node = fit_variant(run, spec["variant"], variant, routed, routed_truth, iterations=spec["iterations"])
        write_json(run / "node.json", node)
        emit("v10_refit_fit_complete", iterations=node["iterations"], rows=len(routed))
        del routed, routed_truth
        ranking = pd.read_parquet(base / "features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking movement identities changed.")
        ranking_mask = route_mask(ranking, "clock_available")
        clock_rows = ranking.loc[ranking_mask].reset_index(drop=True)
        values = node_prediction(run, node, clock_rows, spec["threads"])
        predictions = pd.DataFrame({ID: clock_rows[ID].to_numpy(), "prediction": values})
        predictions.to_parquet(run / "ranking_predictions.parquet", index=False, compression="zstd")
        unusual = clock_rows["nmclock_proxy_s"].lt(0) | clock_rows["nmclock_proxy_s"].gt(10800) | clock_rows["lirf_missing_record"].eq(1)
        indexes = np.unique(np.r_[np.linspace(0, len(clock_rows) - 1, 1000, dtype=int), np.flatnonzero(unusual.to_numpy())])
        restored = node_prediction(run, read_json(run / "node.json"), clock_rows.iloc[indexes].reset_index(drop=True), spec["threads"])
        delta = float(np.max(np.abs(values[indexes] - restored)))
        if delta > 1e-8:
            raise ValueError("The saved cat_long model does not reproduce.")
    report = {"created_at_utc": utc_now(), "run_id": run.name, "stage": "refit", "model_kind": variant["kind"], "variant": spec["variant"],
              "iterations": node["iterations"], "newly_fitted_models": 1, "bundle_id": context["manifest"]["bundle_id"],
              "parent_bundle_id": original["bundle_id"], "model_code": refit_code_receipt(root), "config_sha256": object_hash(config),
              "selection_sha256": sha256_file(selection_path), "selected_candidate": context["selection"]["selected_candidate"],
              "clock_fit_rows": spec["expected_clock_fit_rows"], "clock_fit_ids_sha256": spec["expected_clock_fit_ids_sha256"],
              "refit_months": config["refit_months"], "audit_used": False, "target_filter": "none", "prediction_postprocess": "none",
              "ranking_rows": int(len(ranking)), "ranking_ids_sha256": identity_hash(ranking[ID]),
              "ranking_clock_rows": int(ranking_mask.sum()), "ranking_clock_ids_sha256": identity_hash(predictions[ID]),
              "reload_sample_rows": int(len(indexes)), "max_model_reload_difference_seconds": delta,
              "environment": model_environment(root), "resources": monitor.stats, "artifacts": receipts(run)}
    write_json(run / "run.json", report)
    emit("v10_refit_complete", run=str(run), ranking_clock_rows=int(ranking_mask.sum()), resources=monitor.stats)
    return run
