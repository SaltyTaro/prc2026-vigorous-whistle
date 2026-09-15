"""Package only the new campaign's allowlisted source and declarations."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("infra/azure/.local/phase5r-code.tar"))
args = parser.parse_args()
files = [*sorted((root/"src/prc2026_phase5r").glob("*.py")),
         *sorted((root/"tools").glob("*phase5r*.py")), root/"requirements-phase5r.txt",
         root/"configs/phase5r_v1.json", root/"configs/features_phase5r_v1.json",
         root/"tests/test_phase5r.py", root/"infra/azure/phase5r.sh"]
output = root/args.output
if output.exists() or not output.resolve().is_relative_to((root/"infra/azure/.local").resolve()):
    raise ValueError("Choose a new archive filename under infra/azure/.local; preserve earlier packages.")
output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(output, "w") as archive:
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Unexpected package input.")
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
           "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "size_bytes": output.stat().st_size,
           "files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
output.with_suffix(".json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"package": output.name, "sha256": receipt["sha256"], "size_bytes": receipt["size_bytes"]}))
