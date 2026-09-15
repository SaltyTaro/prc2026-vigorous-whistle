"""Bind refitting to the completed selection and protected movement identities."""
import numpy as np

from prc2026.artifacts import object_hash, read_json, sha256_file
from prc2026.features import ID, TIME
from .inference import SELECTED, WEIGHTS


def refit_assignment(catalog, protocol, months):
    if months != [f"2025-{month:02}" for month in range(1, 12)] or set(months) & set(protocol["reserved_audit_months"]):
        raise ValueError("Only the declared January-November refit is permitted.")
    movement_months = catalog[TIME].dt.strftime("%Y-%m")
    protected = set(catalog.loc[movement_months.isin(protocol["reserved_audit_months"]), "FLIGHT_ID_mvt"].dropna())
    assignment = catalog[[ID]].copy()
    assignment["partition"] = np.where(movement_months.isin(months) & ~catalog["FLIGHT_ID_mvt"].isin(protected), "fit", "reserved")
    return assignment


def eligible_recipe(config, selection, timeline, manifest, code):
    if (selection["audit_used"] or selection["selected_candidate"] != SELECTED
            or selection["config_sha256"] != object_hash(config)
            or selection["bundle_id"] != manifest["bundle_id"]
            or selection["model_code_sha256"] != code["sha256"]
            or selection["native_fit_count"] != 20 or selection["candidate_count"] != 4
            or len(selection["candidates"]) != 4 or set(selection["development_runs"]) != {"seasonal", "forward"}):
        raise ValueError("The unchanged complete eligible selection is required.")
    candidate = selection["candidates"][SELECTED]
    routes = {"clock_available": WEIGHTS, "clock_missing": {"v6": 1.}}
    if (candidate["family"] != "full_timeline" or candidate["routes"] != routes
            or candidate["routes"] != config["candidate_families"]["full_timeline"]
            or candidate["new_weight"] != 1. or not candidate["eligible"]
            or set(candidate["folds"]) != {"seasonal", "forward"}):
        raise ValueError("The selected weights or complementary clock routes changed.")
    for fold, row in candidate["folds"].items():
        ratio = row["metrics"]["overall"]["rmse_seconds"] / selection["reference_metrics"][fold]["overall"]["rmse_seconds"]
        if not np.isfinite(ratio) or ratio > 1 - config["minimum_improvement_fraction_each_fold"]:
            raise ValueError("The selected candidate fails the two-fold promotion margin.")
    eligible = [name for name, row in selection["candidates"].items() if row["eligible"]]
    if min(eligible, key=lambda name: (selection["candidates"][name]["mean_relative_rmse"], name)) != SELECTED:
        raise ValueError("The declared selection order changed.")
    folds = config["development_folds"]
    expected = {"airports": {airport: int(np.rint(np.median([selection["development_runs"][fold]["iterations"][airport] for fold in folds]))) for airport in config["airports"]},
                "globals": {name: int(np.rint(np.median([timeline["development_runs"][name][fold]["iterations"] for fold in folds]))) for name in ["cat_combined", "lgb_combined"]}}
    if selection["refit_iterations"] != expected or min([*expected["airports"].values(), *expected["globals"].values()]) < 1:
        raise ValueError("Refit counts must be the ties-to-even rounded October medians.")
    return candidate


def require_verifications(root, selection_path, selection, code):
    reports = {}
    for name in ["verification.json", "local_verification.json"]:
        path = root / "docs/surface_airports" / name
        report = read_json(path)
        if (not report["passed"] or report["audit_used"]
                or report["selection_sha256"] != sha256_file(selection_path)
                or report["model_code_sha256"] != code["sha256"]
                or report["bundle_id"] != selection["bundle_id"]
                or report["selected_candidate"] != selection["selected_candidate"]
                or report["refit_iterations"] != selection["refit_iterations"]
                or report["native_fit_count"] != 20 or report["candidate_count"] != 4):
            raise ValueError("Matching successful Azure and Windows development verifications are required.")
        reports[name] = sha256_file(path)
    return reports
