"""Small Azure code update; reuse phase-1 data and its verified bundle."""

import json
from pathlib import Path
import tarfile
import hashlib


root = Path(__file__).resolve().parents[1]
files = [*sorted((root / "src/prc2026_models").glob("*.py")),
         *sorted((root / "src/prc2026_submission").glob("*.py")),
         *[root / "tests" / name for name in ["conftest.py", "test_artifacts.py", "test_features.py",
                                              "test_validation.py", "test_models.py", "test_submission.py"]],
         root / "requirements-phase2.txt", root / "tools/run_phase2.py",
         root / "configs/baselines_v1.json", root / "infra/azure/phase2.sh"]
output = root / "infra/azure/.local/phase2-code.tar"
output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(output, "w") as archive:
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError("Unexpected package input.")
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
receipt = {"sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
           "size_bytes": output.stat().st_size,
           "files": [p.relative_to(root).as_posix() for p in files]}
output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"package": output.name, "sha256": receipt["sha256"], "size_bytes": receipt["size_bytes"]}))
