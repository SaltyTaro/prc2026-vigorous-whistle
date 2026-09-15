"""Archive this audit as a source extension and a separate licensed map bundle."""

from pathlib import Path
import hashlib
import json
import re
import sys
import tarfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json


def build_archive(path, members):
    if path.exists():
        raise ValueError("Preserve existing archive: " + str(path))
    files = {}
    for member in sorted(set(members)):
        resolved = member.resolve()
        if not resolved.is_relative_to(ROOT) or not member.is_file() or member.is_symlink():
            raise ValueError("Archive member is not a regular workspace file.")
        name = member.relative_to(ROOT).as_posix()
        if ".env" in member.parts or member.suffix in {".parquet", ".cbm", ".pkl", ".joblib", ".pyc"}:
            raise ValueError("Unexpected competition input, credential or model in source archive.")
        files[name] = sha256_file(member)
    with tarfile.open(path, "x:gz") as archive:
        for name in files:
            archive.add(ROOT / name, arcname=name, recursive=False)
    with tarfile.open(path, "r:gz") as archive:
        entries = archive.getmembers()
        if len(entries) != len(files) or any(not entry.isfile() for entry in entries):
            raise ValueError("Archive members differ.")
        for entry in entries:
            stream = archive.extractfile(entry)
            digest = hashlib.sha256(stream.read()).hexdigest()
            if digest != files[entry.name]:
                raise ValueError("Archive readback differs: " + entry.name)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size, "file_count": len(files), "files": files,
            "every_member_readback_matches": True}


def main():
    output = ROOT / "docs/source_geometry/archive_receipts.json"
    if output.exists():
        raise ValueError("Preserve the original archive receipt.")
    verification = read_json(ROOT / "docs/source_geometry/verification.json")
    if not verification["passed"]:
        raise ValueError("Verification must pass before archiving.")
    for name, expected in verification["input_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Verified evidence changed: " + name)
    docs = ROOT / "docs/source_geometry"
    all_docs = [path for path in docs.rglob("*") if path.is_file()]
    derived = [path for path in all_docs if path.name.endswith("_locations.json") or path.name == "parking_endpoint_controls.json"]
    map_members = [path for path in (ROOT / "data/external/osm/source_geometry_v1").iterdir() if path.is_file()]
    map_members += derived + list((docs / "sources").glob("*.json")) + [docs / "EXTERNAL_DATA_LICENSES.md"]
    tool_names = ["audit_source_reporting.py", "audit_airport_geometry.py", "pull_source_geometry.py",
        "pull_source_geometry_review.py", "review_airport_geometry.py", "review_parking_endpoints.py",
        "summarize_source_geometry.py", "plot_source_geometry.py", "plot_source_geometry_readable.py",
        "plot_source_geometry_readiness.py", "verify_source_geometry.py", "package_source_geometry.py",
        "research_prc_v7_gains.py"]
    source_members = [path for path in all_docs if path not in derived]
    source_members += list((ROOT / "src/prc2026_source_geometry").glob("*.py"))
    source_members += list((ROOT / "configs").glob("source_geometry*_v1.json"))
    source_members += [ROOT / "tools" / name for name in tool_names]
    source_members += [ROOT / "tests/test_source_geometry_audit.py", ROOT / "requirements-source-geometry-audit.txt",
                       ROOT / "README.md"]
    source_members += [path for path in (ROOT / "docs/competition/gain_review_v7_20260915").iterdir() if path.is_file()]
    map_archive = build_archive(ROOT / "runs/source-geometry-audit-map-data.tar.gz", map_members)
    source_archive = build_archive(ROOT / "runs/source-geometry-audit-source.tar.gz", source_members)
    report = {"created_at_utc": utc_now(), "source_and_evidence": source_archive, "map_data": map_archive,
        "source_scope": "New audit source, configuration, focused tests and evidence; an extension of the frozen project, not a replacement of earlier source/model archives.",
        "map_data_license": "Copyright OpenStreetMap contributors, ODbL 1.0; archived parking-tag documentation CC BY-SA 2.0.",
        "audit_used": False, "models_fitted": 0, "predictions_changed": False,
        "verification_sha256": sha256_file(ROOT / "docs/source_geometry/verification.json"),
        "source_inventory_sha256": sha256_file(ROOT / "docs/source_geometry/source_inventory.json"),
        "code_sha256": sha256_file(Path(__file__)), "contains_credentials": False,
        "contains_raw_competition_parquet_or_models": False}
    write_json(output, report)
    links = 0
    for path in [docs / "README.md", docs / "EXTERNAL_DATA_LICENSES.md"]:
        for destination in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if destination.startswith(("http://", "https://", "#")):
                continue
            target = (path.parent / unquote(destination.split("#", 1)[0])).resolve()
            if not target.exists():
                raise ValueError("Documentation link is missing: " + destination)
            links += 1
    completion = {"created_at_utc": utc_now(), "status": "completed", "passed": True,
        "official_best_unchanged": {"version":7, "rmse_seconds":275.0570, "rows":344841},
        "source_audit_complete": True, "geometry_audit_complete": True,
        "next_experiment_recommendation": "LIRF geometry comparison against v7; LFPG historical parking-point coverage remains unresolved.",
        "new_native_models":0, "new_submissions":0, "december_audit_used":False, "azure_actions_performed":False,
        "markdown_links_checked":links, "archive_receipts_sha256":sha256_file(output),
        "verification_sha256":report["verification_sha256"],
        "notes":"No model-score improvement is claimed by these diagnostic results."}
    write_json(docs / "completion.json", completion)
    print(json.dumps({"archives": [{key: value for key, value in entry.items() if key != "files"}
        for entry in [source_archive, map_archive]], "markdown_links_checked":links, "completed":True}, indent=2))


if __name__ == "__main__":
    main()
