"""Preserve the completed Rome comparison without refitting or accessing labels."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from urllib.parse import unquote
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/rome_geometry"
sys.path.insert(0, str(ROOT / "src"))
from prc2026_rome_geometry.common import model_code_receipt


def now():
    return datetime.now(timezone.utc).isoformat()


def safe(relative):
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
        raise ValueError(f"Path escaped the project: {relative}")
    return path


def sha(relative):
    with safe(relative).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(relative):
    return json.loads(safe(relative).read_text(encoding="utf-8"))


def write_new(relative, value):
    with safe(relative).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def check_files(files, snapshots=None):
    snapshots = snapshots or {}
    for name, expected in files.items():
        actual_path = snapshots.get(name, name)
        if sha(actual_path) != expected:
            raise ValueError(f"Preserved file changed: {name} ({actual_path})")
    return len(files)


def check_archive(relative, expected, files):
    if sha(relative) != expected:
        raise ValueError(f"Archive digest mismatch: {relative}")
    with tarfile.open(safe(relative), "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(files) or len({m.name for m in members}) != len(files):
            raise ValueError("Archive member identities differ.")
        for member in members:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts
                    or "\\" in member.name or member.name not in files):
                raise ValueError(f"Unexpected archive member: {member.name}")
            with archive.extractfile(member) as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != files[member.name]:
                raise ValueError(f"Archive readback mismatch: {member.name}")


def make_archive(relative, paths):
    files = {name: sha(name) for name in sorted(set(paths))}
    for name in files:
        parts = PurePosixPath(name).parts
        if any(part in {".env", ".git", ".venv", "__pycache__"} for part in parts):
            raise ValueError(f"Unapproved archive input: {name}")
    with tarfile.open(safe(relative), "x:gz", compresslevel=6) as archive:
        for name in files:
            archive.add(safe(name), arcname=name, recursive=False)
    digest = sha(relative)
    check_archive(relative, digest, files)
    check_files(files)
    return {"path": relative, "sha256": digest, "size_bytes": safe(relative).stat().st_size,
            "file_count": len(files), "files": files, "every_member_readback_matches": True}


def relative_files(directory):
    return [p.relative_to(ROOT).as_posix() for p in safe(directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]


def main():
    outputs = ["narrative_updates.json", "tests_combined.json", "preservation_final.json",
               "archive_receipts.json", "campaign_summary.json"]
    if any((DOC / name).exists() for name in outputs):
        raise ValueError("Completion records already exist; never overwrite them.")
    if any(safe(name).exists() for name in [
            "runs/rome-geometry-source-final.tar.gz", "runs/rome-geometry-feature-extension.tar.gz"]):
        raise ValueError("A final archive already exists; never replace it.")
    result = read("docs/rome_geometry/development_results.json")
    if result["selected_candidate"] is not None or result["refit_iterations"] is not None:
        raise ValueError("This completion workflow is for the non-qualifying batch.")
    if any(c["eligible"] for c in result["candidates"].values()):
        raise ValueError("An eligible candidate needs the separately declared refit workflow.")
    if result["native_fit_count"] != 6 or result["candidate_count"] != 4:
        raise ValueError("Incomplete declared comparison.")
    code = model_code_receipt(ROOT)
    if code["sha256"] != result["model_code_sha256"]:
        raise ValueError("Frozen model source changed.")

    before = read("docs/rome_geometry/narrative_before.json")
    narratives = []
    for item in before["files"]:
        if sha(item["snapshot"]) != item["before_sha256"]:
            raise ValueError("Narrative snapshot changed.")
        narratives.append({**item, "after_sha256": sha(item["path"])})
    narrative_report = {"created_at_utc": now(), "files": narratives}
    snapshots = {item["path"]: item["snapshot"] for item in narratives}

    identities = {}
    for platform, name in [("windows", "tests_final_local.xml"), ("azure", "tests_azure.xml")]:
        cases = ET.parse(DOC / name).findall(".//testcase")
        if any(c.find(tag) is not None for c in cases for tag in ["failure", "error", "skipped"]):
            raise ValueError("A final test did not pass.")
        identities[platform] = sorted((c.attrib["classname"], c.attrib["name"]) for c in cases)
        if len(cases) != 61 or len(set(identities[platform])) != 61:
            raise ValueError("Unexpected final test identities.")
    if identities["windows"] != identities["azure"]:
        raise ValueError("Platform test identities differ.")
    test_report = {"created_at_utc": now(), "passed": True, "distinct_tests_per_platform": 61,
                   "identities": identities["windows"], "reports": {
                       name: sha(f"docs/rome_geometry/{name}")
                       for name in ["tests_final_local.xml", "tests_azure.xml"]}}

    prior = read("docs/next_lever/preservation_final.json")
    raw = {f"data/raw/{name}": digest for name, digest in prior["original_input_hashes"].items()}
    check_files(raw)  # Byte hashes only, including the reserved December file.
    submissions = {f"runs/submissions/vigorous-whistle_{name}.parquet": digest
                   for name, digest in prior["previous_submission_hashes"].items()}
    official = read("docs/surface_refit/submission_v7_official.json")
    submissions[f"runs/submissions/{official['filename']}"] = official["sha256"]
    check_files(submissions)
    check_files(official["artifact_sha256"])

    original_source = read("docs/next_lever/final_source_package.json")
    old_narrative = read("docs/surface_refit/submission_documentation_20260914.json")
    old_snapshots = {i["path"]: i["prior_snapshot"] for i in old_narrative["updated_narratives"]}
    check_files(original_source["files"], old_snapshots)
    for item in old_narrative["updated_narratives"]:
        if item["path"] not in {"README.md"} and sha(item["path"]) != item["after_sha256"]:
            raise ValueError("A previous submission narrative changed.")
    audit_narratives = read("docs/source_geometry/narrative_updates.json")
    for item in audit_narratives["files"]:
        if sha(item["snapshot"]) != item["before_sha256"] or sha(snapshots[item["path"]]) != item["after_sha256"]:
            raise ValueError("Narrative preservation chain changed.")
    check_archive(original_source["archive"], original_source["sha256"], original_source["files"])
    audit_archives = read("docs/source_geometry/archive_receipts.json")
    for kind in ["source_and_evidence", "map_data"]:
        receipt = audit_archives[kind]
        check_files(receipt["files"], snapshots if kind == "source_and_evidence" else None)
        check_archive(receipt["path"], receipt["sha256"], receipt["files"])

    initial = read("docs/rome_geometry/initial_package.json")
    check_files(initial["files"], {"docs/rome_geometry/README.md": snapshots["docs/rome_geometry/README.md"]})
    check_archive(initial["archive"], initial["sha256"], initial["files"])
    stream = read("docs/rome_geometry/development_archive_stream.json")
    mirror = read("docs/rome_geometry/development_mirror.json")
    check_files(stream["files"], mirror["destinations"])
    check_archive(mirror["archive"], stream["sha256"], stream["files"])

    bundle = "data/rome_geometry/587c526337697ab57602"
    manifest = read(f"{bundle}/manifest.json")
    if sha(f"{bundle}/manifest.json") != result["feature_manifest_sha256"]:
        raise ValueError("Frozen feature manifest changed.")
    feature_files = {f"{bundle}/{a['path']}": a["sha256"] for a in manifest["artifacts"]}
    check_files(feature_files)
    verification_paths = [f"docs/rome_geometry/{name}.json" for name in [
        "local_verification", "verification", "local_feature_verification", "feature_verification"]]
    for path in verification_paths:
        report = read(path)
        if not report["passed"] or report["audit_used"]:
            raise ValueError("A final verifier did not pass with the audit protected.")
    native_counts = {}
    for platform, name in [("windows", "local_verification"), ("azure", "verification")]:
        report = read(f"docs/rome_geometry/{name}.json")
        values = [v for fold in report["native_predictions"].values() for v in fold.values()]
        native_counts[platform] = sum(v["rows"] for v in values)
        if native_counts[platform] != 112221 or any(v["maximum_difference_seconds"] != 0 for v in values):
            raise ValueError("Native reconstruction differs.")
    state = read("docs/rome_geometry/final_execution_state.json")
    if state["active_project_python_processes"] or not state["batch_complete"]:
        raise ValueError("The remote batch is not complete.")

    write_new("docs/rome_geometry/narrative_updates.json", narrative_report)
    write_new("docs/rome_geometry/tests_combined.json", test_report)
    preservation = {"created_at_utc": now(), "passed": True, "audit_used": False,
        "scope": "Byte preservation and existing verification records; no label decoding, new fit or scoring.",
        "original_input_sha256": raw, "submission_sha256": submissions,
        "prior_official_artifacts_checked": len(official["artifact_sha256"]),
        "original_source_files_unchanged": len(original_source["files"]) - len(old_snapshots),
        "original_narrative_snapshots_checked": len(old_snapshots),
        "previous_geometry_source_files_checked": audit_archives["source_and_evidence"]["file_count"],
        "previous_geometry_map_files_checked": audit_archives["map_data"]["file_count"],
        "initial_package_files_checked": initial["file_count"],
        "development_archive_and_mirror_files_checked": len(stream["files"]),
        "feature_artifact_sha256": feature_files, "feature_manifest_sha256": result["feature_manifest_sha256"],
        "model_code_sha256": code["sha256"], "frozen_model_source_files_checked": len(code["files"]),
        "distinct_tests_per_platform": 61, "native_prediction_rows_each_platform": native_counts,
        "verification_sha256": {p: sha(p) for p in verification_paths},
        "narrative_updates_sha256": sha("docs/rome_geometry/narrative_updates.json"),
        "refit_performed": False, "export_created": False, "submission_uploaded": False}
    write_new("docs/rome_geometry/preservation_final.json", preservation)

    source_paths = relative_files("src/prc2026_rome_geometry") + relative_files("docs/rome_geometry")
    source_paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / "tools").glob("*rome_geometry*.py")]
    source_paths += ["configs/features_rome_geometry_v1.json", "configs/rome_geometry_v1.json",
                     "tests/test_rome_geometry.py", "requirements-rome-geometry.txt", "README.md"]
    if any(PurePosixPath(p).suffix in {".parquet", ".cbm"} for p in source_paths):
        raise ValueError("Data or model artifact entered the source archive.")
    source_archive = make_archive("runs/rome-geometry-source-final.tar.gz", source_paths)
    feature_paths = relative_files("data/rome_geometry")
    feature_paths += [v["path"] for v in manifest["map_sources"].values()]
    feature_paths += ["docs/source_geometry/EXTERNAL_DATA_LICENSES.md",
                      "docs/source_geometry/source_inventory.json"]
    feature_paths += [p for p in audit_archives["map_data"]["files"]
                      if p.startswith("docs/source_geometry/sources/") or p.endswith(".html")]
    feature_archive = make_archive("runs/rome-geometry-feature-extension.tar.gz", feature_paths)
    archives = {"created_at_utc": now(), "passed": True,
        "source_and_evidence": source_archive, "feature_extension_and_map_inputs": feature_archive,
        "source_scope": "New source, configuration, tests and evidence extending the preserved project. Excludes this archive receipt and the subsequent completion record to avoid circular hashes.",
        "feature_scope": "New ID-aligned LIRF feature columns, route catalog and cached public maps. No original competition input or December label file.",
        "external_data_license": "Copyright OpenStreetMap contributors, ODbL 1.0 for maps and derived geometry; parking-tag documentation CC BY-SA 2.0. Competition-derived features retain their separate input-data constraints.",
        "parent_source_archives": {original_source["archive"]: original_source["sha256"],
                                   audit_archives["source_and_evidence"]["path"]: audit_archives["source_and_evidence"]["sha256"]},
        "parent_map_archive": {"path": audit_archives["map_data"]["path"],
                               "sha256": audit_archives["map_data"]["sha256"]},
        "initial_package": {"path": initial["archive"], "sha256": initial["sha256"]},
        "development_archive": {"path": mirror["archive"], "sha256": mirror["sha256"],
                                "file_count": len(stream["files"])},
        "contains_credentials": False, "audit_used": False}
    write_new("docs/rome_geometry/archive_receipts.json", archives)

    links = 0
    for name in ["README.md", "docs/rome_geometry/README.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", safe(name).read_text(encoding="utf-8")):
            target = target.split("#", 1)[0].strip("<>")
            if not target or re.match(r"[a-zA-Z]+://", target):
                continue
            if not (safe(name).parent / unquote(target)).resolve().exists():
                raise ValueError(f"Broken local documentation link in {name}: {target}")
            links += 1
    completion = {"created_at_utc": now(), "passed": True, "status": "complete_no_candidate_promoted",
        "selected_candidate": None, "native_fit_count": 6, "candidate_count": 4,
        "current_official_version": 7, "current_official_rmse_seconds": official["official_rmse_seconds"],
        "official_result_is_a_saved_snapshot": True, "audit_used": False,
        "refit_performed": False, "export_created": False, "submission_uploaded": False,
        "additional_workers_started": False, "local_markdown_links_checked": links,
        "artifact_sha256": {p: sha(p) for p in [
            "docs/rome_geometry/README.md", "docs/rome_geometry/analysis.json",
            "docs/rome_geometry/development_results.json", "docs/rome_geometry/tests_combined.json",
            "docs/rome_geometry/preservation_final.json", "docs/rome_geometry/narrative_updates.json",
            "docs/rome_geometry/archive_receipts.json", "docs/rome_geometry/final_execution_state.json",
            "docs/surface_refit/submission_v7_official.json", *verification_paths]}}
    write_new("docs/rome_geometry/campaign_summary.json", completion)
    print(json.dumps({"passed": True, "decision": completion["status"],
        "raw_files_preserved": len(raw), "submissions_preserved": len(submissions),
        "source_archive": {k: v for k, v in source_archive.items() if k != "files"},
        "feature_archive": {k: v for k, v in feature_archive.items() if k != "files"},
        "local_markdown_links_checked": links}, indent=2))


if __name__ == "__main__":
    main()
