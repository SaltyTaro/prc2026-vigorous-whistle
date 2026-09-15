"""Verify an Azure archive and restore Phase-5 results to local mirror paths."""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import sha256_file


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    archive_path = root / args.archive
    if sha256_file(archive_path) != args.expected_sha256.lower():
        raise ValueError("Transferred archive differs from its producing-machine hash.")
    mappings = {
        "runs/phase5/": root / "runs/azure_arcw0_phase5",
        "runs/phase5_followup/": root / "runs/azure_arcw0_phase5_followup",
        "docs/phase5/": root / "docs/phase5",
    }
    logs = {"runs/" + name for name in ["phase5-console.log", "phase5-remaining-console.log", "phase5-refit-console.log", "phase5-verification-console.log"]}
    restored, identical = 0, 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            source = PurePosixPath(member.name)
            if not member.isfile() or source.is_absolute() or ".." in source.parts or "\\" in member.name:
                raise ValueError("Unexpected archive entry.")
            if member.name in logs:
                base = root / "runs/azure_arcw0_phase5_logs"
                destination = base / source.name
            else:
                prefix = next((p for p in mappings if member.name.startswith(p)), None)
                if prefix is None:
                    raise ValueError("Archive contains a path outside the Phase-5 allowlist.")
                base = mappings[prefix]
                destination = base / Path(*PurePosixPath(member.name[len(prefix):]).parts)
            if not destination.resolve().is_relative_to(base.resolve()):
                raise ValueError("Archive path escapes its local mirror.")
            data = archive.extractfile(member).read()
            if destination.exists():
                if sha256_file(destination) != hashlib.sha256(data).hexdigest():
                    raise ValueError(f"Preserve the differing existing artifact: {destination.name}")
                identical += 1
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(data)
                restored += 1
    print({"archive_sha256": args.expected_sha256.lower(), "restored_files": restored, "identical_existing_files": identical})


if __name__ == "__main__":
    main()
