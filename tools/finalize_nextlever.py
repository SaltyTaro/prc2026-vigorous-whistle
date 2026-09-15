"""Bind the completed search, immutable refit, local export and preservation."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_ladder.training import model_code_receipt as ladder_code
from prc2026_ladder_scale.training import model_code_receipt as scale_code
from prc2026_ladder_blend.recipes import model_code_receipt as blend_code
from prc2026_timeline.training import model_code_receipt as timeline_code
from prc2026_arrivalctx.training import model_code_receipt as arrival_code
from prc2026_surface_airports.training import model_code_receipt as airport_code
from prc2026_surface_refit.training import model_code_receipt as refit_code
from summarize_nextlever_development import same_science


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=int)
    args = parser.parse_args()
    output = ROOT / "docs/next_lever/campaign_summary.json"
    preservation_path = ROOT / "docs/next_lever/preservation_final.json"
    if output.exists() or preservation_path.exists():
        raise ValueError("Preserve the final campaign evidence.")
    baseline = read_json(ROOT / "docs/next_lever/preservation_during_training.json")
    for name, digest in baseline["original_input_hashes"].items():
        if sha256_file(ROOT / "data/raw" / name) != digest:
            raise ValueError("A supplied original input changed: " + name)
    for name, digest in baseline["submission_hashes"].items():
        if sha256_file(ROOT / "runs/submissions" / ("vigorous-whistle_" + name + ".parquet")) != digest:
            raise ValueError("A previous version changed: " + name)
    expected = {
        "ladder": (ladder_code, "137bd8c306d044887cbcb6fdd673cddead97c5d81c5156c0d8a1ade6c820f6e6"),
        "ladder_scale": (scale_code, "2c5ee07eea80711b07704e4c68d552161073802456101ea026be8cabb236fe53"),
        "ladder_blend": (blend_code, "c064573f92c623ffa9dbb6f9fd23e6365aa3ad51eed06d4d6fb0df27bd367e51"),
        "timeline": (timeline_code, "3f7f148faf5a4b4b33382b2b66837c007e5520b1abe3718e37f99487977ed584"),
        "arrivalctx": (arrival_code, "af910d80897bd0f07b5e5a312471ecbfdd3803b825f67d5b0736f96f730a1a13"),
        "surface_airports": (airport_code, "0fe8598a2ac9e5921907cd5fb9f0679f14e1d11be91bc5f28f73a6610b74673c"),
        "surface_refit": (refit_code, "4391d8dc1647f32b2175737bf546f72b5418328f4ac14c069319d6bc9827874b"),
    }
    source_hashes = {}
    for name, (reader, digest) in expected.items():
        if reader(ROOT)["sha256"] != digest:
            raise ValueError("A frozen source package changed: " + name)
        source_hashes[name] = digest
    tests = read_json(ROOT / "docs/next_lever/tests_combined.json")
    for name, digest in tests["test_source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("A current test source changed after its combined passing record.")
    for name, digest in tests["evidence_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Recorded passing test evidence changed.")
    development = read_json(ROOT / "docs/next_lever/development_summary.json")
    for name, digest in development["evidence_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("A completed development artifact changed.")
    ready_path = ROOT / "docs/surface_refit" / f"submission_v{args.version}.json"
    ready = read_json(ready_path)
    if ready["status"] != "validated_export_ready" or ready["submitted"] or ready["audit_used"]:
        raise ValueError("A verified local export with no automatic submission is required.")
    if sha256_file(ROOT / "runs/submissions" / ready["filename"]) != ready["sha256"]:
        raise ValueError("The new export changed.")
    azure, windows = [read_json(ROOT / "docs/surface_refit" / name) for name in ["verification.json", "local_verification.json"]]
    same_science(azure, windows)
    if not azure["passed"] or not windows["passed"] or azure["run_sha256"] != ready["refit_run_sha256"]:
        raise ValueError("Both final verifiers must refer to this exported refit.")
    run = ROOT / "runs/azure_arcw0_surface_refit" / ready["refit_run_id"]
    refit = read_json(run / "run.json")
    mirror = read_json(ROOT / "docs/surface_refit/refit_mirror.json")
    if not mirror["passed"] or sha256_file(ROOT / mirror["archive"]) != mirror["sha256"]:
        raise ValueError("The refit archive differs from its source-machine stream receipt.")
    state = read_json(ROOT / "docs/surface_refit/final_execution_state.json")
    if state["campaign_training_or_evaluation_processes"] or state["host"] != "arcw0" or state["new_workers_started"] != 0:
        raise ValueError("The bounded refit and verification processes must finish.")
    preservation = {"created_at_utc": utc_now(), "passed": True, "audit_used": False,
                    "original_input_hashes": baseline["original_input_hashes"], "previous_submission_hashes": baseline["submission_hashes"],
                    "new_local_export": {"filename": ready["filename"], "sha256": ready["sha256"]},
                    "source_hashes": source_hashes, "distinct_tests_per_platform": tests["distinct_current_tests"],
                    "scope": "All fourteen original input hashes, v1-v6 bytes, frozen scientific source packages and recorded tests. No December labels were scored and no remote bucket write was made."}
    write_json(preservation_path, preservation)
    evidence = {name: sha256_file(ROOT / name) for name in [
        "docs/next_lever/development_summary.json", "docs/next_lever/tests_combined.json", "docs/next_lever/preservation_final.json",
        "docs/surface_refit/declaration.json", "docs/surface_refit/verification.json", "docs/surface_refit/local_verification.json",
        "docs/surface_refit/refit_mirror.json", "docs/surface_refit/final_execution_state.json", "docs/surface_refit/azure_allocation_check.json",
        ready_path.relative_to(ROOT).as_posix()]}
    report = {"created_at_utc": utc_now(), "complete": True, "audit_used": False,
              "development_native_fits": 32, "additional_preserved_timeline_fit_attempts": 1,
              "declared_candidate_recipes": 22, "selected_candidate": development["selected_candidate"],
              "new_refit_native_models": 12, "new_refit_trees": refit["iterations"],
              "refit_run_id": run.name, "refit_run_sha256": ready["refit_run_sha256"],
              "refit_resources": refit["resources"], "complete_ranking_rows_verified_each_platform": azure["complete_ranking_rows"],
              "complete_native_reload_maximum_difference_seconds": azure["complete_native_reload_maximum_difference_seconds"],
              "independent_weight_maximum_difference_seconds": azure["independent_weight_maximum_difference_seconds"],
              "unchanged_missing_clock_rows": azure["fallback_rows_identical_to_v6"],
              "local_export": {"filename": ready["filename"], "sha256": ready["sha256"], "rows": ready["rows"], "submitted": False},
              "official_best": ready["official_best_unchanged"], "all_platform_verifications_passed": True,
              "distinct_tests_passed_each_platform": tests["distinct_current_tests"], "evidence_sha256": evidence,
              "decision": "Selected surface context and airport extension refitted, independently verified and exported; ready for a separate official submission.",
              "code_sha256": sha256_file(Path(__file__))}
    write_json(output, report)
    print({key: report[key] for key in ["complete", "selected_candidate", "refit_run_id", "local_export", "all_platform_verifications_passed"]})


if __name__ == "__main__":
    main()
