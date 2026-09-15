"""Package a credential-free Phase-4 code update for the existing Azure worker."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
files = [*sorted((root / "src/prc2026_phase4").glob("*.py")),
         root / "requirements-phase4.txt", root / "configs/features_phase4_v1.json", root / "configs/phase4_v1.json",
         root / "tools/prepare_phase4.py", root / "tools/run_phase4.py", root / "tools/select_phase4.py",
         root / "tools/verify_phase4.py", root / "tests/test_phase4.py", root / "infra/azure/phase4.sh"]
output = root / "infra/azure/.local/phase4-code.tar"
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
