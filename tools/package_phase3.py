"""Package a small Phase-3 code update for the existing arcw0 project."""

import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
files = [*sorted((root / "src/prc2026_phase3").glob("*.py")),
         root / "requirements-phase3.txt", root / "configs/features_v1.json",
         root / "configs/phase3_v1.json", root / "tools/prepare_phase3.py", root / "tools/run_phase3.py",
         root / "tests/test_phase3_features.py", root / "tests/test_phase3_models.py",
         root / "infra/azure/phase3.sh", root / "infra/azure/phase3_followup.sh",
         root / "tools/select_phase3.py", root / "tools/verify_phase3.py", root / "configs/phase3_followup_v1.json"]
output = root / "infra/azure/.local/phase3-code.tar"
output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(output, "w") as archive:
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Unexpected package input.")
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
receipt = {"sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "size_bytes": output.stat().st_size,
           "files": [p.relative_to(root).as_posix() for p in files]}
output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"package": output.name, "sha256": receipt["sha256"], "size_bytes": receipt["size_bytes"]}))
