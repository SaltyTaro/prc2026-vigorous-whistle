"""Run the final bounded airport-feature comparison on the existing arcw0."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import socket
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    if socket.gethostname() != "arcw0" or ROOT != Path("/home/azureuser/prc-flight-2026"):
        raise RuntimeError("Only the authorized isolated arcw0 workspace can launch this batch.")
    destination = ROOT / "docs/surface_airports/batch_completion.json"
    if destination.exists():
        raise RuntimeError("Preserve the existing batch completion receipt.")
    for name in ["timeline", "arrivalctx"]:
        if not json.loads((ROOT / "docs" / name / "verification.json").read_text())["passed"]:
            raise RuntimeError("Finish verification of preceding experiments first.")
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        if any(part in {"tools/run_timeline.py", "tools/run_arrivalctx.py", "tools/run_surface_airports.py"} for part in command):
            raise RuntimeError("Another competition training process is still active.")
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = "16"
    steps = [["-m", "pytest", "-q", "--junitxml=docs/surface_airports/tests_azure.xml"],
             ["tools/run_surface_airports.py", "--fold", "seasonal"],
             ["tools/run_surface_airports.py", "--fold", "forward"],
             ["tools/select_surface_airports.py"], ["tools/verify_surface_airports.py"],
             ["tools/summarize_surface_airports.py"]]
    records = []
    for arguments in steps:
        row = {"arguments": arguments, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        result = subprocess.run([sys.executable, "-u", "-B", *arguments], cwd=ROOT, check=False)
        row.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode)
        records.append(row)
        if result.returncode:
            break
    report = {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "steps": records,
              "declared_batch_complete": len(records) == len(steps) and all(row["returncode"] == 0 for row in records),
              "audit_used": False, "submission_uploaded": False}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not report["declared_batch_complete"]:
        raise SystemExit(records[-1]["returncode"])


if __name__ == "__main__":
    main()
