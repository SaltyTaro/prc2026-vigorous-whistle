"""Close the completed ladder batches from saved receipts, without reading labels."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def save(relative, document):
    with (ROOT / relative).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def test_evidence(paths):
    cases = set()
    for relative in paths:
        tree = ET.parse(ROOT / relative)
        for case in tree.findall(".//testcase"):
            assert all(case.find(tag) is None for tag in ("failure", "error", "skipped"))
            cases.add((case.attrib["classname"], case.attrib["name"]))
    return cases, {
        "distinct_passing_tests": len(cases),
        "reports": {path: digest(ROOT / path) for path in paths},
    }


def main():
    from prc2026_ladder.training import model_code_receipt as ladder_code
    from prc2026_ladder_scale.training import model_code_receipt as scale_code
    from prc2026_ladder_blend.recipes import model_code_receipt as blend_code

    now = datetime.now(timezone.utc).isoformat()
    state = read("docs/ladder/final_execution_state.json")
    assert not state["campaign_training_or_evaluation_processes"]
    assert state["host"] == "arcw0" and state["new_workers_started"] == 0
    codes = {
        "ladder": ladder_code(ROOT)["sha256"],
        "ladder_scale": scale_code(ROOT)["sha256"],
        "ladder_blend": blend_code(ROOT)["sha256"],
    }
    assert codes == state["frozen_code_hashes"]
    assert read("docs/ladder/final_submission_integrity.json")["passed"]

    windows_cases, windows_tests = test_evidence([
        "docs/ladder_scale/tests_local.xml",
        "docs/ladder_scale/selection_tests_local.xml",
        "docs/ladder_scale/tests_review_local.xml",
        "docs/ladder_blend/tests_local.xml",
    ])
    azure_cases, azure_tests = test_evidence(["docs/ladder_blend/tests_azure.xml"])
    assert windows_cases == azure_cases and len(windows_cases) == 153

    blend = read("docs/ladder_blend/development_results.json")
    selection_hash = digest(ROOT / "docs/ladder_blend/development_results.json")
    azure = read("docs/ladder_blend/verification.json")
    windows = read("docs/ladder_blend/local_verification.json")
    azure_fields = {key: value for key, value in azure.items() if key != "created_at_utc"}
    windows_fields = {key: value for key, value in windows.items() if key != "created_at_utc"}
    assert azure_fields == windows_fields
    assert azure["passed"] and not azure["audit_used"]
    assert azure["selection_sha256"] == selection_hash
    assert azure["selected_candidate"] is None
    assert azure["model_code_sha256"] == codes["ladder_blend"]
    native_max = max(
        model["maximum_difference_seconds"]
        for fold in azure["native_parent_checks"].values()
        for family in fold.values()
        for model in family.values()
    )
    assert native_max == 0
    comparison = {
        "created_at_utc": now,
        "passed": True,
        "compared_fields": sorted(azure_fields),
        "all_scientific_fields_equal": True,
        "selection_sha256": selection_hash,
        "azure_verification_sha256": digest(ROOT / "docs/ladder_blend/verification.json"),
        "windows_verification_sha256": digest(ROOT / "docs/ladder_blend/local_verification.json"),
        "maximum_native_prediction_difference_seconds": native_max,
        "audit_used": False,
    }
    save("docs/ladder_blend/verification_comparison.json", comparison)

    reference = {
        fold: metrics["overall"]["rmse_seconds"]
        for fold, metrics in blend["reference_metrics"].items()
    }
    candidates = []
    for name, candidate in blend["candidates"].items():
        scores = {
            fold: item["metrics"]["overall"]["rmse_seconds"]
            for fold, item in candidate["folds"].items()
        }
        improvements = {fold: 1 - score / reference[fold] for fold, score in scores.items()}
        eligible = min(improvements.values()) >= 0.005
        assert eligible == candidate["eligible"]
        candidates.append({
            "name": name,
            "rmse_seconds": scores,
            "improvement_fraction": improvements,
            "mean_relative_rmse": candidate["mean_relative_rmse"],
            "eligible": eligible,
        })
    candidates.sort(key=lambda item: (item["mean_relative_rmse"], item["name"]))
    assert len(candidates) == 8 and not any(item["eligible"] for item in candidates)
    nearest = max(candidates, key=lambda item: min(item["improvement_fraction"].values()))
    both_count = sum(min(item["improvement_fraction"].values()) > 0 for item in candidates)
    save("docs/ladder_blend/analysis.json", {
        "created_at_utc": now,
        "code_sha256": digest(Path(__file__)),
        "selection_sha256": selection_hash,
        "candidate_count": len(candidates),
        "candidates_by_mean_relative_rmse": candidates,
        "leading_mean_ratio_candidate": candidates[0]["name"],
        "closest_to_joint_margin": nearest,
        "closest_rule": "Largest minimum fold improvement; descriptive only, not a selection rule.",
        "candidates_improving_both_folds": both_count,
        "eligible_candidates": 0,
        "selected_candidate": None,
        "decision": "retain v6",
        "new_labels_read": False,
        "audit_used": False,
    })
    save("docs/ladder_blend/completion.json", {
        "created_at_utc": now,
        "complete": True,
        "development_native_fits": 0,
        "candidate_recipes": 8,
        "candidates_improving_both_folds": both_count,
        "eligible_candidates": 0,
        "selected_candidate": None,
        "decision": "retain v6",
        "both_platform_verifications_passed": True,
        "selection_sha256": selection_hash,
        "verification_comparison_sha256": digest(ROOT / "docs/ladder_blend/verification_comparison.json"),
        "archive_sha256": read("docs/ladder_blend/development_archive.json")["sha256"],
        "refit_created": False,
        "submission_exported": False,
        "submission_uploaded": False,
        "audit_used": False,
    })

    batches = []
    all_candidates = []
    wall = cpu = 0.0
    peak = 0
    evidence = {}
    for name, fits, expected_count in [("ladder", 8, 36), ("ladder_scale", 8, 20), ("ladder_blend", 0, 8)]:
        completion = read(f"docs/{name}/completion.json")
        analysis = read(f"docs/{name}/analysis.json")
        verify = read(f"docs/{name}/verification_comparison.json")
        assert completion["complete"] and completion["selected_candidate"] is None
        assert not completion["audit_used"] and not completion["submission_uploaded"]
        assert verify["passed"]
        assert verify["maximum_native_prediction_difference_seconds"] == 0
        assert analysis["candidate_count"] == expected_count
        current = analysis["candidates_by_mean_relative_rmse"]
        assert not any(item["eligible"] for item in current)
        all_candidates.extend({"campaign": name, **item} for item in current)
        batches.append({
            "campaign": name,
            "development_native_fits": fits,
            "candidate_recipes": expected_count,
            "eligible_candidates": 0,
            "leading_mean_ratio_candidate": current[0],
            "selection_sha256": completion["selection_sha256"],
            "both_platform_verifications_passed": True,
        })
        if fits:
            resources = analysis["fit_resources"]
            wall += resources["total_wall_seconds"]
            cpu += resources["total_cpu_seconds"]
            peak = max(peak, resources["maximum_sampled_rss_bytes"])
        for filename in ("completion.json", "verification_comparison.json", "development_archive.json"):
            relative = f"docs/{name}/{filename}"
            evidence[relative] = digest(ROOT / relative)
    nearest_all = max(all_candidates, key=lambda item: min(item["improvement_fraction"].values()))
    assert nearest_all["name"] == nearest["name"]
    standings = read("docs/ladder/leaderboard_after_campaign.json")
    assert standings["team_best"]["filename"] == "vigorous-whistle_v6.parquet"
    assert standings["team_best"]["score"] == 277.4508
    for relative in ["docs/ladder/final_execution_state.json", "docs/ladder/final_submission_integrity.json", "docs/ladder/leaderboard_after_campaign.json"]:
        evidence[relative] = digest(ROOT / relative)
    summary = {
        "created_at_utc": now,
        "complete": True,
        "code_sha256": digest(Path(__file__)),
        "decision": "retain v6; no candidate meets the declared two-fold promotion margin",
        "development_native_fits": 16,
        "declared_candidate_recipes": len(all_candidates),
        "eligible_candidates": 0,
        "minimum_improvement_fraction_each_fold": 0.005,
        "reference_development_rmse_seconds": reference,
        "closest_to_joint_margin": nearest_all,
        "closest_rule": "Largest minimum fold improvement; descriptive only, not a selection rule.",
        "batches": batches,
        "tests": {"windows": windows_tests, "azure": azure_tests, "identical_test_identities": True},
        "all_platform_verifications_passed": True,
        "maximum_native_prediction_difference_seconds": 0.0,
        "maximum_independent_scale_difference_seconds": 0.0,
        "fit_resources": {"total_wall_seconds": wall, "total_cpu_seconds": cpu, "maximum_sampled_rss_bytes": peak},
        "frozen_model_code_sha256": codes,
        "official_result": standings["team_best"],
        "leaderboard": {key: standings[key] for key in ("created_at_utc", "team_rank", "scored_teams", "rmse_reduction_fraction_to_third")},
        "third_place_rmse_seconds": standings["top_three"][2]["score"],
        "refit_created": False,
        "submission_exported": False,
        "submission_uploaded": False,
        "audit_used": False,
        "compute": {"worker": "arcw0", "new_workers_started": 0, "active_campaign_processes": [], "vm_left_running": True},
        "evidence_sha256": evidence,
        "interpretation": "Reused complete development folds, distinct from official scores. All three declared batches are closed. Further interpolation was not evaluated; the promotion margin remains unchanged.",
    }
    assert summary["declared_candidate_recipes"] == 64
    save("docs/ladder/campaign_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("complete", "decision", "development_native_fits", "declared_candidate_recipes", "closest_to_joint_margin")}))


if __name__ == "__main__":
    main()
