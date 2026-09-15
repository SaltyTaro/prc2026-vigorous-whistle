"""Archive only this campaign's verified artifacts for a local evidence mirror."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, verify_receipts, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/phase5r"))
    parser.add_argument("--include-features", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = []
    if not args.features_only:
        runs = root/args.runs
        if runs.resolve() != (root/"runs/phase5r").resolve():
            raise ValueError("Only the Phase-5R run directory may be archived.")
        for path in sorted(runs.glob("*/run.json")):
            verify_receipts(path.parent, read_json(path)["artifacts"])
        for base in [runs, root/"docs/phase5r"]:
            files.extend(p for p in base.rglob("*") if p.is_file())
    if args.include_features or args.features_only:
        latest = root/"data/phase5r/latest.json"
        bundle = root/read_json(latest)["bundle"]
        if not bundle.resolve().is_relative_to((root/"data/phase5r").resolve()):
            raise ValueError("Unexpected feature bundle location.")
        manifest = read_json(bundle/"manifest.json")
        verify_receipts(bundle, manifest["artifacts"])
        files += [latest, bundle/"manifest.json", *[bundle/item["path"] for item in manifest["artifacts"]]]
    destination = root/args.output
    if destination.exists() or not destination.resolve().is_relative_to((root/"runs").resolve()):
        raise ValueError("Use a new archive filename under runs; existing evidence is immutable.")
    with tarfile.open(destination, "w:gz", compresslevel=3) as archive:
        for path in sorted(set(files)):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive input.")
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "path": destination.relative_to(root).as_posix(),
               "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size, "file_count": len(set(files)),
               "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(set(files))}}
    write_json(destination.with_suffix(destination.suffix+".json"), receipt)
    print({k: receipt[k] for k in ["path", "sha256", "size_bytes", "file_count"]})


if __name__ == "__main__":
    main()
