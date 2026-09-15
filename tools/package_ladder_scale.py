"""Package the separate target-scale campaign and the new first-ladder audit helpers."""
import argparse
from pathlib import Path
import sys
import tarfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_ladder_scale.training import model_code_receipt

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("infra/azure/.local/ladder-scale-code-initial.tar.gz"))
args = parser.parse_args()
declaration = read_json(root/"docs/ladder_scale/declaration.json")
if (sha256_file(root/"configs/ladder_scale_v1.json") != declaration["configuration_sha256"]
        or model_code_receipt(root)["sha256"] != declaration["model_code_sha256"]):
    raise ValueError("The declared configuration or model code changed.")
files = [*sorted((root/"src/prc2026_ladder_scale").glob("*.py")),
         *sorted((root/"tools").glob("*ladder_scale*.py")),
         *sorted((root/"tests").glob("test_ladder_scale*.py")),
         root/"configs/ladder_scale_v1.json", root/"requirements-ladder-scale.txt",
         root/"infra/azure/ladder_scale.sh", root/"docs/ladder_scale/README.md", root/"docs/ladder_scale/declaration.json"]
files += [root/"tools"/name for name in [
    "summarize_ladder.py", "audit_ladder_controls.py", "audit_ladder_clock_context.py", "audit_ladder_fitting_regimes.py"]]
files += [root/"docs/ladder"/name for name in ["clock_context_coverage.json", "fitting_clock_regimes.json"]]
destination = root/args.output
if destination.exists() or not destination.resolve().is_relative_to((root/"infra/azure/.local").resolve()):
    raise ValueError("Use a fresh archive within infra/azure/.local.")
destination.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(destination, "w:gz") as archive:
    for path in sorted(set(files)):
        if not path.is_file() or path.is_symlink():
            raise ValueError("Unexpected package input.")
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
receipt = {
    "created_at_utc": utc_now(), "path": destination.relative_to(root).as_posix(),
    "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size,
    "model_code_sha256": model_code_receipt(root)["sha256"],
    "files": {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(set(files))},
}
write_json(destination.with_suffix(".json"), receipt)
print({key: receipt[key] for key in ["path", "sha256", "size_bytes", "model_code_sha256"]})

