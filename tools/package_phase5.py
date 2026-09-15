"""Package only the Phase-5 source/configuration update for the existing worker."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
files = [*sorted((root / "src/prc2026_phase5").glob("*.py")),
         *sorted((root / "tools").glob("*phase5*.py")),
         *sorted((root / "infra/azure").glob("phase5*.sh")),
         root / "requirements-phase5.txt", *sorted((root / "configs").glob("phase5*.json")), root / "tests/test_phase5.py"]
output = root / "infra/azure/.local/phase5-code.tar"
output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(output, "w") as archive:
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Unexpected code-package input.")
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
receipt = {"sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "size_bytes": output.stat().st_size,
           "files": [p.relative_to(root).as_posix() for p in files]}
output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"package": output.name, "sha256": receipt["sha256"], "size_bytes": receipt["size_bytes"]}))
