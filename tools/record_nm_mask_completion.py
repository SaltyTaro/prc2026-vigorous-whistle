"""Preserve the completed parallel comparison without another fit or score."""
from pathlib import Path
import xml.etree.ElementTree as ET

from record_rome_geometry_completion import (
    ROOT, now, safe, sha, read, write_new, check_files, check_archive,
    make_archive, relative_files,
)
from prc2026_nm_mask.common import code_receipt


DOC = "docs/nm_mask/"


def test_cases(path, count):
    cases = ET.parse(safe(path)).findall(".//testcase")
    identities = sorted((case.attrib["classname"], case.attrib["name"]) for case in cases)
    if len(cases) != count or len(set(identities)) != count:
        raise ValueError("Unexpected test identities: " + path)
    if any(case.find(tag) is not None for case in cases for tag in ["failure", "error", "skipped"]):
        raise ValueError("A final test did not pass: " + path)
    return identities


def main():
    outputs = ["narrative_updates.json", "tests_combined.json", "preservation_final.json",
               "campaign_summary.json", "archive_receipts.json"]
    if any(safe(DOC + name).exists() for name in outputs):
        raise ValueError("Final campaign evidence already exists; preserve it.")
    result = read(DOC + "development_results.json")
    if (result["native_fit_count"] != 8 or result["candidate_count"] != 12
            or result["selected_candidate"] is not None or result["refit_iterations"] is not None
            or any(item["eligible"] for item in result["candidates"].values())):
        raise ValueError("This closure requires the complete non-promoted batch.")
    source = code_receipt(ROOT)
    declaration = read(DOC + "declaration_models.json")
    if source != declaration["model_code"] or source["sha256"] != result["model_code_sha256"]:
        raise ValueError("Frozen model source changed.")
    if sha("src/prc2026_nm_mask/data.py") != declaration["job_input_code_sha256"]:
        raise ValueError("Frozen job loading/preparation source changed.")
    check_files(declaration["artifact_sha256"])

    verification_paths = [DOC + name for name in [
        "local_feature_verification.json", "local_verification.json", "verification.json"]]
    for path in verification_paths:
        report = read(path)
        if not report["passed"] or report["audit_used"]:
            raise ValueError("Verification is incomplete: " + path)
    native = {}
    for platform, name in [("windows", "local_verification.json"), ("azure", "verification.json")]:
        report = read(DOC + name)
        if (report["native_prediction_rows"] != 26916 or report["complete_candidate_fold_metrics_reproduced"] != 24
                or report["selected_candidate"] is not None or report["model_code_sha256"] != source["sha256"]
                or report["selection_sha256"] != sha(DOC + "development_results.json")
                or report["verifier_code_sha256"] != sha("tools/verify_nm_mask.py")
                or any(v["maximum_difference_seconds"] != 0 for v in report["native_predictions"].values())):
            raise ValueError("A native verification no longer matches the frozen results.")
        native[platform] = report["native_prediction_rows"]

    initial = test_cases(DOC + "tests_local.xml", 34)
    tests = {DOC + "tests_local.xml": sha(DOC + "tests_local.xml")}
    mirrors = []
    jobs = read(DOC + "jobs_bound.json")["jobs"]
    all_job_files = {}
    for item in jobs:
        job = read(item["path"])
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("A frozen job definition changed.")
        all_job_files.update(job["files"])
        mirror = read(DOC + "mirrors/" + item["worker"] + ".json")
        if not mirror["passed"] or not mirror["every_member_readback_matches"]:
            raise ValueError("A worker mirror is incomplete.")
        check_files(mirror["files"], mirror["destinations"])
        check_archive(mirror["archive"], mirror["sha256"], mirror["files"])
        test_name = ("tests_" if item["worker"] == "arcw0" else "tests_resumed_") + item["job_id"] + ".xml"
        test_path = mirror["destinations"][DOC + test_name]
        if test_cases(test_path, 34) != initial:
            raise ValueError("Worker test identities differ: " + item["worker"])
        tests[test_path] = sha(test_path)
        mirrors.append({"worker": item["worker"], "archive": mirror["archive"], "sha256": mirror["sha256"],
                        "file_count": len(mirror["files"]), "size_bytes": mirror["size_bytes"]})
    check_files(all_job_files)
    regression_paths = [DOC + name for name in ["tests_verifier_local.xml", "tests_verifier_azure.xml"]]
    regression = [test_cases(path, 4) for path in regression_paths]
    if regression[0] != regression[1]:
        raise ValueError("Verifier regression tests differ across platforms.")
    tests.update({path: sha(path) for path in regression_paths})
    test_report = {"created_at_utc": now(), "passed": True, "campaign_tests_per_worker": 34,
        "workers": [item["worker"] for item in jobs], "campaign_tests_local": 34,
        "verifier_regression_tests_windows_and_arcw4": 4, "campaign_test_identities": initial,
        "verifier_regression_identities": regression[0], "report_sha256": tests}

    # Resolve changed narrative files only through explicitly recorded snapshots.
    snapshots = {}
    for item in read("docs/surface_refit/submission_documentation_20260914.json")["updated_narratives"]:
        snapshots.setdefault(item["path"], []).append((item["prior_snapshot"], item["before_sha256"]))
    for path in ["docs/source_geometry/narrative_updates.json", "docs/rome_geometry/narrative_updates.json"]:
        for item in read(path)["files"]:
            snapshots.setdefault(item["path"], []).append((item["snapshot"], item["before_sha256"]))
    before = read(DOC + "narrative_before.json")
    narratives = []
    for item in before["files"]:
        snapshots.setdefault(item["path"], []).append((item["snapshot"], item["sha256"]))
        narratives.append({"path": item["path"], "snapshot": item["snapshot"],
                           "before_sha256": item["sha256"], "after_sha256": sha(item["path"])})
    for candidates in snapshots.values():
        for snapshot, digest in candidates:
            if sha(snapshot) != digest:
                raise ValueError("A historical narrative snapshot changed.")
    def previous_files(files):
        replacements = {}
        for path, digest in files.items():
            if sha(path) == digest:
                continue
            matching = [snapshot for snapshot, expected in snapshots.get(path, []) if expected == digest]
            if not matching:
                raise ValueError("An earlier scientific artifact changed: " + path)
            replacements[path] = matching[0]
        check_files(files, replacements)
        return len(files)

    prior = read("docs/next_lever/preservation_final.json")
    raw = {"data/raw/" + name: digest for name, digest in prior["original_input_hashes"].items()}
    check_files(raw)  # Byte hashing only, including the reserved December file.
    official = read("docs/surface_refit/submission_v7_official.json")
    submissions = {"runs/submissions/vigorous-whistle_" + version + ".parquet": digest
                   for version, digest in prior["previous_submission_hashes"].items()}
    submissions["runs/submissions/" + official["filename"]] = official["sha256"]
    check_files(submissions)
    check_files(official["artifact_sha256"])
    old = read("docs/next_lever/final_source_package.json")
    older_archives = [{"path": old["archive"], "sha256": old["sha256"], "files": old["files"]}]
    geometry = read("docs/source_geometry/archive_receipts.json")
    older_archives.extend(geometry[key] for key in ["source_and_evidence", "map_data"])
    rome = read("docs/rome_geometry/archive_receipts.json")
    older_archives.extend(rome[key] for key in ["source_and_evidence", "feature_extension_and_map_inputs"])
    older_counts = {}
    for archive in older_archives:
        older_counts[archive["path"]] = previous_files(archive["files"])
        check_archive(archive["path"], archive["sha256"], archive["files"])

    bundle = read("data/nm_mask/latest.json")["bundle"]
    manifest = read(bundle + "/manifest.json")
    feature_files = {bundle + "/" + entry["path"]: entry["sha256"] for entry in manifest["artifacts"]}
    check_files(feature_files)
    if sha(bundle + "/manifest.json") != read(DOC + "local_feature_verification.json")["manifest_sha256"]:
        raise ValueError("Frozen feature manifest changed.")
    packages = []
    package_paths = [DOC + "packages/" + item["job_id"] + ".json" for item in jobs]
    package_paths.append(DOC + "verification_package.json")
    for path in package_paths:
        package = read(path)
        contents = read(package["manifest"])["files"]
        contents = {**contents, package["manifest"]: sha(package["manifest"])}
        check_files(contents)
        check_archive(package["archive"], package["sha256"], contents)
        packages.append({"path": package["archive"], "sha256": package["sha256"], "file_count": len(contents)})
    candidate_files = {"runs/azure_parallel_nm_mask/" + path: digest
                       for path, digest in result["candidate_prediction_sha256"].items()}
    check_files(candidate_files)
    states = read(DOC + "pool_final_state.json")["workers_before"]
    if any(row["powerState"] != ("VM running" if row["name"] == "arcw0" else "VM deallocated") for row in states):
        raise ValueError("The worker pool has not returned to its initial allocation state.")
    checks = read(DOC + "pool_final_checks.json")
    if len(checks["workers"]) != 8 or any(row["active_project_python_processes"] for row in checks["workers"]):
        raise ValueError("The completed workers must have no project Python jobs.")

    write_new(DOC + "narrative_updates.json", {"created_at_utc": now(), "files": narratives})
    write_new(DOC + "tests_combined.json", test_report)
    preservation = {"created_at_utc": now(), "passed": True, "audit_used": False,
        "scope": "Byte preservation, archive readback and existing verification; no new model fit or score.",
        "original_input_sha256": raw, "submission_sha256": submissions,
        "official_v7_artifacts_checked": len(official["artifact_sha256"]),
        "older_archive_file_counts": older_counts, "feature_artifact_sha256": feature_files,
        "feature_manifest_sha256": sha(bundle + "/manifest.json"), "model_code_sha256": source["sha256"],
        "job_input_code_sha256": declaration["job_input_code_sha256"], "job_input_files_checked": len(all_job_files),
        "worker_archives": mirrors, "input_packages": packages,
        "complete_candidate_sha256": candidate_files, "native_rows_each_platform": native,
        "verification_sha256": {path: sha(path) for path in verification_paths},
        "refit_performed": False, "export_created": False, "submission_uploaded": False}
    write_new(DOC + "preservation_final.json", preservation)
    analysis = read(DOC + "analysis.json")
    resource_values = [item["resources"] for item in result["development_runs"].values()]
    summary = {"created_at_utc": now(), "completed": True, "native_fits": 8, "fixed_candidates": 12,
        "best_by_mean_relative_rmse": analysis["best_by_mean_relative_rmse"], "selected_candidate": None,
        "candidate_scores": analysis["candidate_scores"], "v7_official_rmse_seconds": 275.0570,
        "sum_native_fit_wall_seconds": sum(value["wall_seconds"] for value in resource_values),
        "largest_native_fit_wall_seconds": max(value["wall_seconds"] for value in resource_values),
        "maximum_sampled_model_rss_bytes": max(value["peak_process_rss_bytes_sampled"] for value in resource_values),
        "workers_used": sorted(item["worker"] for item in jobs), "final_vm_states": states,
        "native_prediction_rows_each_platform": native, "audit_used": False,
        "refit_performed": False, "export_created": False, "submission_uploaded": False,
        "model_code_sha256": source["sha256"], "selection_sha256": sha(DOC + "development_results.json"),
        "preservation_sha256": sha(DOC + "preservation_final.json"),
        "archived_runner_correction": DOC + "launcher_fix.json", "archived_verifier_correction": DOC + "verifier_fix/correction.json"}
    write_new(DOC + "campaign_summary.json", summary)
    source_paths = relative_files("src/prc2026_nm_mask") + relative_files("docs/nm_mask")
    source_paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / "tools").glob("*nm_mask*.py")]
    source_paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("test_nm_mask*.py")]
    source_paths += ["tools/prc_pool.py", "configs/nm_mask_v1.json", "configs/features_nm_mask_v1.json",
        "requirements-nm-mask.txt", "infra/azure/parallel_authorization_20260915.json", "infra/azure/README.md",
        "docs/competition/AZURE_COMPUTE.md", "README.md"]
    source_archive = make_archive("runs/nm-mask-source-final.tar.gz", source_paths)
    features_archive = make_archive("runs/nm-mask-feature-bundle.tar.gz", relative_files(bundle) + ["data/nm_mask/latest.json"])
    write_new(DOC + "archive_receipts.json", {"created_at_utc": now(), "passed": True,
        "source_and_evidence": source_archive, "feature_bundle": features_archive,
        "portable_training_and_verification_inputs": packages, "worker_results": mirrors,
        "parent_archives": [{k: v for k, v in archive.items() if k != "files"} for archive in older_archives],
        "complete_candidate_sha256": candidate_files, "contains_credentials": False,
        "feature_bundle_contains_labels": False, "audit_used": False})
    print({"completed": True, "selected_candidate": None, "source_sha256": source_archive["sha256"],
           "feature_archive_sha256": features_archive["sha256"], "final_states_verified": True}, flush=True)


if __name__ == "__main__":
    main()
