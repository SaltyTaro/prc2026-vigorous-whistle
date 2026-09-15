"""Resume the unchanged declaration after the recorded provenance-import fix."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if socket.gethostname() != "arcw0" or ROOT != Path("/home/azureuser/prc-flight-2026"):
        raise RuntimeError("Only the existing isolated arcw0 workspace is authorized.")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = "16"
    steps = [
        ["-m", "pytest", "-q", "--junitxml=docs/timeline/import_fix/tests_azure.xml"],
        ["tools/verify_timeline.py", "--features-only"],
        ["tools/run_timeline.py", "--fold", "seasonal"],
        ["tools/run_timeline.py", "--fold", "forward"],
        ["tools/select_timeline.py"],
        ["tools/verify_timeline.py"],
    ]
    records = []
    for arguments in steps:
        record = {"arguments": arguments, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        result = subprocess.run([sys.executable, "-u", "-B", *arguments], cwd=ROOT, check=False)
        record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode)
        records.append(record)
        if result.returncode:
            break
    report = {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "steps": records,
              "declared_batch_complete": len(records) == len(steps) and all(r["returncode"] == 0 for r in records),
              "audit_used": False, "submission_uploaded": False}
    with (ROOT/"docs/timeline/import_fix/resumed_completion.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not report["declared_batch_complete"]:
        raise SystemExit(records[-1]["returncode"])


if __name__ == "__main__":
    main()
