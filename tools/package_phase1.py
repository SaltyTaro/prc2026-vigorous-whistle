"""Package an explicit allowlist for the Azure pilot; never include credentials."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=root / "infra/azure/.local/phase1-pilot.tar")
    args = parser.parse_args()
    files = [
        *sorted((root / "src/prc2026").glob("*.py")),
        *[root / "tests" / name for name in [
            "conftest.py", "test_artifacts.py", "test_features.py", "test_validation.py"]],
        *sorted((root / "data/raw").glob("*.parquet")),
        root / "tools/run_phase1.py", root / "pyproject.toml",
        root / "requirements-phase1.txt", root / "configs/validation_v1.json",
        root / "docs/competition/download_manifest.json",
        root / "infra/azure/pilot.sh", root / "infra/azure/allocation.json",
    ]
    if len(list((root / "data/raw").glob("*.parquet"))) != 14:
        raise ValueError("Expected all 14 audited input files.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w") as archive:
        for path in files:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected package input: {path.name}")
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    digest = hashlib.sha256()
    with args.output.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    receipt = {"package": args.output.name, "sha256": digest.hexdigest(),
               "size_bytes": args.output.stat().st_size,
               "files": [path.relative_to(root).as_posix() for path in files]}
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))


if __name__ == "__main__":
    main()
