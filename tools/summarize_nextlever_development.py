"""Consolidate completed declarations and independent platform verification."""
from copy import deepcopy
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json


def same_science(left, right, path=""):
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("Verification fields differ at " + path)
        deltas = [same_science(left[key], right[key], path + "/" + key) for key in left if key != "created_at_utc"]
        return max(deltas, default=0.)
    if isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise ValueError("Verification list differs at " + path)
        return max((same_science(a, b, path + "/" + str(i)) for i, (a, b) in enumerate(zip(left, right))), default=0.)
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or isinstance(right, bool) or not math.isclose(left, right, rel_tol=0., abs_tol=1e-9):
            raise ValueError("Verification numeric value differs at " + path)
        return abs(left - right)
    if left != right:
        raise ValueError("Verification value differs at " + path)
    return 0.


def native_differences(value):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (int, float)) and "max" in key and "difference" in key:
                found.append(child)
            else:
                found.extend(native_differences(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(native_differences(child))
    return found


def main():
    destination = ROOT / "docs/next_lever/development_summary.json"
    if destination.exists():
        raise ValueError("Preserve the completed development summary.")
    campaigns = [("timeline", 8, 10), ("arrivalctx", 4, 6), ("nextlever_joint", 0, 2), ("surface_airports", 20, 4)]
    batches, evidence = [], {}
    for name, fits, recipes in campaigns:
        base = ROOT / "docs" / name
        selection_path = base / "development_results.json"
        selection = read_json(selection_path)
        azure, windows = [read_json(base / filename) for filename in ["verification.json", "local_verification.json"]]
        if selection["audit_used"] or selection["candidate_count"] != recipes or len(selection["candidates"]) != recipes:
            raise ValueError("A complete fixed recipe declaration changed.")
        for report in [azure, windows]:
            if not report["passed"] or report["audit_used"] or report["selection_sha256"] != sha256_file(selection_path) or report["selected_candidate"] != selection["selected_candidate"]:
                raise ValueError("A platform report does not verify the complete selection.")
        left, right = deepcopy(azure), deepcopy(windows)
        if name == "surface_airports":
            feature_paths = [ROOT / "docs/timeline/feature_verification.json", ROOT / "docs/timeline/local_feature_verification.json"]
            for report, path in zip([left, right], feature_paths):
                if report.pop("feature_verification_sha256") != sha256_file(path):
                    raise ValueError("An independent feature verification reference changed.")
            same_science(*[read_json(path) for path in feature_paths])
        numeric_delta = same_science(left, right)
        differences = native_differences(azure) + native_differences(windows)
        if not differences or max(differences) != 0.:
            raise ValueError("Native reconstruction must be exact on both platforms.")
        ordered = sorted(selection["candidates"], key=lambda key: (selection["candidates"][key]["mean_relative_rmse"], key))
        table = []
        for candidate in ordered:
            row = selection["candidates"][candidate]
            table.append({"name": candidate, "eligible": row["eligible"],
                          "rmse_seconds": {fold: metrics["metrics"]["overall"]["rmse_seconds"] for fold, metrics in row["folds"].items()},
                          "improvement_fraction": {fold: 1 - metrics["metrics"]["overall"]["rmse_seconds"] / selection["reference_metrics"][fold]["overall"]["rmse_seconds"] for fold, metrics in row["folds"].items()}})
        mirror = read_json(base / "development_mirror.json")
        archive_path = ROOT / mirror["archive"]
        if not mirror["passed"] or sha256_file(archive_path) != mirror["sha256"]:
            raise ValueError("An immutable development archive differs from its mirror receipt.")
        batches.append({"campaign": name, "development_native_fits": fits, "candidate_recipes": recipes,
                        "selected_candidate": selection["selected_candidate"], "eligible_candidates": len([row for row in table if row["eligible"]]),
                        "candidates_by_mean_relative_rmse": table, "selection_sha256": sha256_file(selection_path),
                        "both_platform_verifications_passed": True, "maximum_native_difference_seconds": 0.,
                        "maximum_platform_numeric_difference": numeric_delta, "archive_sha256": mirror["sha256"],
                        "archive_size_bytes": mirror["size_bytes"]})
        for filename in ["development_results.json", "verification.json", "local_verification.json", "development_mirror.json"]:
            path = base / filename
            evidence[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    tests = read_json(ROOT / "docs/next_lever/tests_combined.json")
    if not tests["passed"] or tests["distinct_current_tests"] != 196:
        raise ValueError("Complete current test coverage is required.")
    timeline_analysis = read_json(ROOT / "docs/timeline/analysis.json")
    arrival_analysis = read_json(ROOT / "docs/arrivalctx/analysis.json")
    airport_analysis = read_json(ROOT / "docs/surface_airports/analysis.json")
    selected = read_json(ROOT / "docs/surface_airports/development_results.json")
    report = {"created_at_utc": utc_now(), "complete": True, "audit_used": False,
              "development_native_fits": sum(row[1] for row in campaigns), "declared_candidate_recipes": sum(row[2] for row in campaigns),
              "additional_preserved_timeline_fit_attempts": 1, "batches": batches,
              "selected_candidate": selected["selected_candidate"], "selected_campaign": "surface_airports",
              "selected_development_diagnostics": airport_analysis["selected_diagnostics"],
              "paired_day_comparison": "Selected surface recipe minus v6; see paired_day_label_note.json for the inherited display label.",
              "selected_paired_days": selected["selected_paired_days"],
              "refit_component_iterations": selected["refit_iterations"], "new_refit_native_models": 12,
              "completed_fit_resources_by_campaign": {"timeline": timeline_analysis["completed_fit_resources"],
                                                     "arrivalctx": arrival_analysis['fit_resources'],
                                                     "surface_airports": airport_analysis["native_fit_resources"]},
              "distinct_tests_passed_each_platform": tests["distinct_current_tests"],
              "all_platform_development_verifications_passed": True, "evidence_sha256": evidence,
              "code_sha256": sha256_file(Path(__file__)),
              "decision": "The full timeline recipe with new airport models qualifies. Refit it at the frozen counts; the other declarations remain rejected.",
              "official_best": {"version": 6, "rmse_seconds": 277.4508, "rows": 344841},
              "scope": "Complete reused development evidence. This report does not claim refit completion, an official new score or use of the December audit."}
    write_json(destination, report)
    print({key: report[key] for key in ["complete", "development_native_fits", "declared_candidate_recipes", "selected_candidate", "distinct_tests_passed_each_platform"]})


if __name__ == "__main__":
    main()
