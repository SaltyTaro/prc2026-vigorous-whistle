"""Finish the declared surface batch after a verifier-only correction."""
from pathlib import Path
import os
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_surface_airports.training import model_code_receipt


def main():
    if socket.gethostname() != "arcw0" or ROOT != Path("/home/azureuser/prc-flight-2026"):
        raise RuntimeError("Only the existing isolated arcw0 workspace is authorized.")
    docs = ROOT / "docs/surface_airports"
    destination = docs / "completion_after_verifier_fix.json"
    if destination.exists():
        raise ValueError("Preserve the recorded verification correction.")
    initial = read_json(docs / "batch_completion.json")
    correction = read_json(docs / "verifier_fix/correction.json")
    if initial["declared_batch_complete"] or len(initial["steps"]) != 5 or any(row["returncode"] for row in initial["steps"][:4]):
        raise ValueError("This correction only resumes the completed, selected development batch.")
    if initial["steps"][-1]["arguments"] != ["tools/verify_surface_airports.py"] or initial["steps"][-1]["returncode"] != 1:
        raise ValueError("Unexpected original failure.")
    if sha256_file(ROOT / "tools/verify_surface_airports.py") != correction["after_sha256"]:
        raise ValueError("The corrected verifier has changed.")
    if model_code_receipt(ROOT)["sha256"] != correction["model_code_sha256_unchanged"]:
        raise ValueError("The frozen model code must remain unchanged.")
    selection_path = ROOT / "runs/surface_airports/selection.json"
    selection_hash = sha256_file(selection_path)
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = "16"
    steps = [["-m", "pytest", "-q", "tests/test_surface_verifier.py", "--junitxml=docs/surface_airports/verifier_fix/tests_azure.xml"],
             ["tools/verify_surface_airports.py"], ["tools/summarize_surface_airports.py"]]
    records = []
    for arguments in steps:
        row = {"arguments": arguments, "started_at_utc": utc_now()}
        result = subprocess.run([sys.executable, "-u", "-B", *arguments], cwd=ROOT, check=False)
        row.update(finished_at_utc=utc_now(), returncode=result.returncode)
        records.append(row)
        if result.returncode:
            break
    if sha256_file(selection_path) != selection_hash:
        raise ValueError("A verification-only correction must not change selection.")
    report = {"finished_at_utc": utc_now(), "steps": records,
              "initial_completion_sha256": sha256_file(docs / "batch_completion.json"),
              "correction_sha256": sha256_file(docs / "verifier_fix/correction.json"),
              "selection_sha256_unchanged": selection_hash, "models_retrained": 0,
              "declared_batch_complete": len(records) == len(steps) and all(row["returncode"] == 0 for row in records),
              "audit_used": False, "submission_uploaded": False}
    write_json(destination, report)
    if not report["declared_batch_complete"]:
        raise SystemExit(records[-1]["returncode"])
    print({"declared_batch_complete": True, "models_retrained": 0}, flush=True)


if __name__ == "__main__":
    main()
