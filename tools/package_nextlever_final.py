"""Archive reviewed reproduction source and evidence, excluding data and secrets."""
from pathlib import Path
import hashlib
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_ladder_scale.training import model_code_receipt as scale_code
from prc2026_ladder_blend.recipes import model_code_receipt as blend_code
from prc2026_arrivalctx.training import model_code_receipt as arrival_code
from prc2026_surface_refit.training import model_code_receipt as refit_code


def main():
    archive_path = ROOT / "runs/nextlever-source-final.tar.gz"
    receipt_path = ROOT / "docs/next_lever/final_source_package.json"
    if archive_path.exists() or receipt_path.exists():
        raise ValueError("Preserve the existing final source archive.")
    summary = read_json(ROOT / "docs/next_lever/campaign_summary.json")
    if not summary["complete"] or not summary["all_platform_verifications_passed"]:
        raise ValueError("Finish the complete selected refit and export first.")
    files = set()
    for reader in [scale_code, blend_code, arrival_code, refit_code]:
        files.update(ROOT / name for name in reader(ROOT)["files"])
    tokens = ["timeline", "arrivalctx", "surface", "nextlever", "next_lever", "clock_relink", "arrival_context"]
    files.update(path for path in (ROOT / "tools").glob("*.py") if any(token in path.name for token in tokens))
    files.update(ROOT / "tools" / name for name in ["select_phase4.py", "verify_phase5.py", "build_submission.py", "prc_data.py", "prc_submit.py"])
    files.update((ROOT / "configs").glob("*.json"))
    files.update((ROOT / "tests").rglob("*.py"))
    files.update(ROOT.glob("requirements-*.txt"))
    for name in ["README.md", "LICENSE", "pytest.ini", "pyproject.toml", ".gitignore", ".gitattributes"]:
        if (ROOT / name).is_file():
            files.add(ROOT / name)
    for name in ["next_lever", "timeline", "arrivalctx", "nextlever_joint", "surface_airports", "surface_refit"]:
        files.update(path for path in (ROOT / "docs" / name).rglob("*") if path.is_file() and "azure_snapshots" not in path.parts
                     and path.suffix in {".json", ".md", ".xml", ".txt", ".html", ".snapshot", ".log"})
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()) or path.name == ".env":
            raise ValueError("A source archive entry escaped the explicit workspace allowlist.")
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in sorted(files)}
    with tarfile.open(archive_path, "x:gz", compresslevel=6) as archive:
        for path in sorted(files):
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(hashes) or {member.name for member in members} != set(hashes):
            raise ValueError("The source archive file set changed.")
        for member in members:
            if not member.isfile() or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != hashes[member.name]:
                raise ValueError("A source archive member differs from its reviewed file.")
    for name, digest in hashes.items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("A source file changed during packaging.")
    report = {"created_at_utc": utc_now(), "passed": True, "archive": archive_path.relative_to(ROOT).as_posix(),
              "sha256": sha256_file(archive_path), "size_bytes": archive_path.stat().st_size,
              "file_count": len(hashes), "files": hashes, "campaign_summary_sha256": sha256_file(ROOT / "docs/next_lever/campaign_summary.json"),
              "scope": "Reproduction source, configs, tests and supporting evidence. No credentials, raw Parquet inputs, trained models or inherited cloud scaffolding. Original data/model archives remain separate.",
              "code_sha256": sha256_file(Path(__file__))}
    write_json(receipt_path, report)
    print({key: report[key] for key in ["passed", "archive", "sha256", "size_bytes", "file_count"]})


if __name__ == "__main__":
    main()
