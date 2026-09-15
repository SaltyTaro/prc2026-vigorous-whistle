"""Allowlisted weather code and verified public covariates; never credentials."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/"src"))
from prc2026.artifacts import read_json, sha256_file, write_json
from prc2026_weather.features import verify_external

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("infra/azure/.local/weather-code-inputs.tar.gz"))
args = parser.parse_args()
external = verify_external(root, read_json(root/"configs/features_weather_v1.json"))
files = [*sorted((root/"src/prc2026_weather").glob("*.py")), *sorted((root/"tools").glob("*weather*.py")),
         root/"requirements-weather.txt", root/"configs/weather_v1.json", root/"configs/features_weather_v1.json",
         *sorted((root/"tests").glob("test_weather*.py")), root/"infra/azure/weather.sh",
         root/"docs/competition/weather_20260914/runway_sources.json"]
for receipt in external.values():
    files += [root/item["path"] for item in receipt["objects"]]
output = root/args.output
if output.exists() or not output.resolve().is_relative_to((root/"infra/azure/.local").resolve()):
    raise ValueError("Choose a new archive under infra/azure/.local; existing packages are preserved.")
output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(output, "w:gz") as archive:
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Unexpected package input.")
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha256_file(output),
           "size_bytes": output.stat().st_size, "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in files}}
write_json(output.with_suffix(".json"), receipt)
print(json.dumps({"package": output.name, "sha256": receipt["sha256"], "size_bytes": receipt["size_bytes"], "files": len(files)}))
