"""Restore a verified ladder archive and retain separate Azure document snapshots."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import sha256_file

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--expected-sha256", required=True)
args = parser.parse_args()
archive_path = root/args.archive
digest = sha256_file(archive_path)
if digest != args.expected_sha256.lower():
    raise ValueError("Transferred archive differs from its producing-machine hash.")
mappings = {"runs/ladder/": root/"runs/azure_arcw0_ladder",
            "docs/ladder/": root/"docs/ladder/azure_snapshots"/digest[:16]}
restored, identical = 0, 0
with tarfile.open(archive_path, "r:gz") as archive:
    for member in archive:
        path = PurePosixPath(member.name)
        if not member.isfile() or path.is_absolute() or ".." in path.parts or "\\" in member.name:
            raise ValueError("Unexpected archive entry.")
        prefix = next((value for value in mappings if member.name.startswith(value)), None)
        if prefix is None:
            raise ValueError("Archive path is outside this campaign.")
        base = mappings[prefix]
        destination = base/Path(*PurePosixPath(member.name[len(prefix):]).parts)
        if not destination.resolve().is_relative_to(base.resolve()):
            raise ValueError("Archive path escapes its local mirror.")
        payload = archive.extractfile(member).read()
        if destination.exists():
            if sha256_file(destination) != hashlib.sha256(payload).hexdigest():
                raise ValueError("Preserve differing local evidence: "+str(destination))
            identical += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
            restored += 1
print({"sha256": digest, "restored_files": restored, "identical_files": identical,
       "azure_document_snapshot": mappings["docs/ladder/"].relative_to(root).as_posix()})
