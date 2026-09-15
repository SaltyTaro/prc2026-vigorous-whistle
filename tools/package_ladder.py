"""Package only new ladder source, tests, declarations and their input receipts."""
import argparse
from pathlib import Path
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_ladder.training import model_code_receipt

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("infra/azure/.local/ladder-code-initial.tar.gz"))
args = parser.parse_args()
declaration = read_json(root/"docs/ladder/declaration.json")
if sha256_file(root/"configs/ladder_v1.json") != declaration["configuration_sha256"]:
    raise ValueError("The initial ladder declaration changed.")
files = [*sorted((root/"src/prc2026_ladder").glob("*.py")),
         *sorted((root/"tools").glob("*ladder*.py")),
         *sorted((root/"tests").glob("test_ladder*.py")),
         root/"configs/ladder_v1.json", root/"requirements-ladder.txt", root/"infra/azure/ladder.sh",
         root/"docs/ladder/README.md", root/"docs/ladder/declaration.json", root/"docs/ladder/headroom_v6.json"]
destination = root/args.output
if destination.exists() or not destination.resolve().is_relative_to((root/"infra/azure/.local").resolve()):
    raise ValueError("Use a fresh archive within infra/azure/.local.")
destination.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(destination, "w:gz") as archive:
    for path in sorted(set(files)):
        if not path.is_file() or path.is_symlink():
            raise ValueError("Unexpected package input.")
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
receipt = {"created_at_utc": utc_now(), "path": destination.relative_to(root).as_posix(),
           "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size,
           "model_code_sha256": model_code_receipt(root)["sha256"],
           "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(set(files))}}
write_json(destination.with_suffix(".json"), receipt)
print({k: receipt[k] for k in ["path", "sha256", "size_bytes", "model_code_sha256"]})
