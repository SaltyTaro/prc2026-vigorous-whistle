"""Run the declared arrival-context batch after the existing timeline job finishes."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def timeline_active():
    result = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
        try:
            command = process.info["cmdline"] or []
            if process.info["cwd"] == str(ROOT) and any(part in {"tools/run_timeline.py", "tools/resume_timeline.py"} for part in command):
                result.append(process.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def main():
    if socket.gethostname() != "arcw0" or ROOT != Path("/home/azureuser/prc-flight-2026"):
        raise RuntimeError("Only the authorized isolated arcw0 workspace can launch this batch.")
    destination = ROOT / "docs/arrivalctx/batch_completion.json"
    if destination.exists():
        raise RuntimeError("Preserve the previous completion receipt.")
    queued_at = now()
    deadline = time.monotonic() + 7200
    while timeline_active():
        if time.monotonic() > deadline:
            raise RuntimeError("The earlier campaign has not finished within the bounded queue period.")
        print(json.dumps({"event": "waiting_for_timeline", "time": now(), "pids": timeline_active()}), flush=True)
        time.sleep(30)
    completion = json.loads((ROOT / "docs/timeline/import_fix/resumed_completion.json").read_text())
    verification = json.loads((ROOT / "docs/timeline/verification.json").read_text())
    if not completion["declared_batch_complete"] or not verification["passed"]:
        raise RuntimeError("Resolve the preceding timeline batch before starting new training.")
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = "16"
    steps = [
        ["-m", "pytest", "-q", "--junitxml=docs/arrivalctx/tests_azure.xml"],
        ["tools/verify_arrivalctx.py", "--features-only"],
        ["tools/run_arrivalctx.py", "--fold", "seasonal"],
        ["tools/run_arrivalctx.py", "--fold", "forward"],
        ["tools/select_arrivalctx.py"],
        ["tools/verify_arrivalctx.py"],
        ["tools/summarize_arrivalctx.py"],
    ]
    records = []
    for arguments in steps:
        record = {"arguments": arguments, "started_at_utc": now()}
        result = subprocess.run([sys.executable, "-u", "-B", *arguments], cwd=ROOT, check=False)
        record.update(finished_at_utc=now(), returncode=result.returncode)
        records.append(record)
        if result.returncode:
            break
    report = {"queued_at_utc": queued_at, "finished_at_utc": now(), "steps": records,
              "declared_batch_complete": len(records) == len(steps) and all(row["returncode"] == 0 for row in records),
              "audit_used": False, "submission_uploaded": False}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not report["declared_batch_complete"]:
        raise SystemExit(records[-1]["returncode"])


if __name__ == "__main__":
    main()
