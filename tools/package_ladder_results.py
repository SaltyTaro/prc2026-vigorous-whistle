"""Archive only ladder runs and their evidence, preserving existing archives."""
import argparse
from pathlib import Path
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, verify_receipts, write_json

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
runs = root/"runs/ladder"
for path in sorted(runs.glob("*/run.json")):
    verify_receipts(path.parent, read_json(path)["artifacts"])
files = [path for base in [runs, root/"docs/ladder"] for path in base.rglob("*") if path.is_file()]
destination = root/args.output
if destination.exists() or not destination.resolve().is_relative_to((root/"runs").resolve()):
    raise ValueError("Use a new archive filename within runs.")
with tarfile.open(destination, "w:gz", compresslevel=3) as archive:
    for path in sorted(set(files)):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unexpected archive input.")
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
receipt = {"created_at_utc": utc_now(), "path": destination.relative_to(root).as_posix(),
           "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size,
           "file_count": len(set(files)),
           "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(set(files))}}
write_json(destination.with_suffix(destination.suffix+".json"), receipt)
print({key: receipt[key] for key in ["path", "sha256", "size_bytes", "file_count"]})
