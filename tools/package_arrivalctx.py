"""Package only declared new arrival inputs and supporting evidence for arcw0."""
from pathlib import Path
import io
import json
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_arrivalctx.training import model_code_receipt


def main():
    destination = ROOT / "infra/azure/.local/arrivalctx-initial.tar.gz"
    receipt_path = ROOT / "docs/arrivalctx/initial_package.json"
    if destination.exists() or receipt_path.exists():
        raise ValueError("Preserve the original arrival input package.")
    bundle = ROOT / read_json(ROOT / "data/arrivalctx/latest.json")["bundle"]
    files = list((ROOT / "src/prc2026_arrivalctx").glob("*.py"))
    names = [
        "configs/arrivalctx_v1.json", "configs/features_arrivalctx_v1.json", "requirements-arrivalctx.txt",
        "data/arrivalctx/latest.json", "tools/prepare_arrivalctx.py", "tools/run_arrivalctx.py",
        "tools/select_arrivalctx.py", "tools/check_arrivalctx_features.py", "tools/verify_arrivalctx.py",
        "tools/summarize_arrivalctx.py", "tools/batch_arrivalctx.py", "tools/package_arrivalctx.py",
        "tools/summarize_timeline.py", "tools/audit_arrival_context.py", "tools/audit_clock_relink.py", "tools/audit_clock_relink_quality.py",
        "tests/test_arrival_context_audit.py", "tests/test_arrivalctx.py", "docs/arrivalctx/declaration.json",
        "docs/arrivalctx/README.md", "docs/arrivalctx/tests_local.xml", "docs/arrivalctx/tests_initial_local.xml",
        "docs/arrivalctx/local_feature_verification.json", "docs/next_lever/arrival_context_audit_declaration.json",
        "docs/next_lever/arrival_context_audit.json", "docs/next_lever/clock_relink_coverage.json", "docs/next_lever/clock_relink_quality.json",
        "runs/research_next_lever/arrival_context_features.parquet", "runs/research_next_lever/clock_relink_candidates.parquet",
        "runs/research_next_lever/clock_relink_reviewed.parquet",
    ]
    files += [ROOT / name for name in names] + [bundle / "manifest.json", bundle / "arrival_features.parquet"]
    files = sorted(set(files))
    inventory = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files}
    if any(not path.resolve().is_relative_to(ROOT) or path.is_symlink() for path in files):
        raise ValueError("Package input escaped its workspace.")
    model_code = model_code_receipt(ROOT)
    inventory_payload = (json.dumps({"files": inventory, "model_code_sha256": model_code["sha256"]}, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "x:gz", compresslevel=3) as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        member = tarfile.TarInfo("docs/arrivalctx/package_inventory.json")
        member.size, member.mode = len(inventory_payload), 0o644
        archive.addfile(member, io.BytesIO(inventory_payload))
    for path in files:
        if sha256_file(path) != inventory[path.relative_to(ROOT).as_posix()]:
            raise ValueError("An input changed while packaging.")
    report = {"created_at_utc": utc_now(), "package": destination.relative_to(ROOT).as_posix(),
              "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size,
              "files": inventory, "file_count": len(files), "model_code": model_code,
              "audit_used": False, "secrets_included": False,
              "scope": "Explicit new experiment files, supplied-arrival derived features, and prior fitting-only research evidence. No environment or credential file is included."}
    write_json(receipt_path, report)
    print({key: report[key] for key in ["package", "sha256", "size_bytes", "file_count"]})


if __name__ == "__main__":
    main()
