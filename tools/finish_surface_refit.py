"""Verify the one running selected refit when it finishes; never retrain it."""
from pathlib import Path
import argparse
import socket
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--process-created", required=True, type=float)
    args = parser.parse_args()
    if socket.gethostname() != "arcw0" or ROOT != Path("/home/azureuser/prc-flight-2026"):
        raise RuntimeError("Use only the existing isolated arcw0 workspace.")
    destination = ROOT / "docs/surface_refit/batch_completion.json"
    if destination.exists():
        raise ValueError("Preserve the recorded refit completion.")
    started, deadline = utc_now(), time.monotonic() + 7200
    while time.monotonic() < deadline:
        try:
            process = psutil.Process(args.pid)
            same_process = process.create_time() == args.process_created and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            same_process = False
        if not same_process:
            break
        time.sleep(10)
    else:
        raise RuntimeError("The bounded refit has exceeded its two-hour verification wait; no process was stopped.")
    runs = [p.parent for p in (ROOT / "runs/surface_refit").glob("*/run.json") if read_json(p)["stage"] == "refit"]
    completed = len(runs) == 1
    report = {"started_at_utc": started, "refit_completed": completed, "pid": args.pid,
              "process_created": args.process_created, "audit_used": False, "submission_uploaded": False,
              "automatic_retraining": False, "native_verification_passed": False}
    if completed:
        report["run_id"] = runs[0].name
        report["run_sha256"] = sha256_file(runs[0] / "run.json")
        result = subprocess.run([sys.executable, "-u", "-B", "tools/verify_surface_refit.py"], cwd=ROOT, check=False)
        report["verification_returncode"] = result.returncode
        report["native_verification_passed"] = result.returncode == 0
        if result.returncode == 0:
            report["verification_sha256"] = sha256_file(ROOT / "docs/surface_refit/verification.json")
    report["finished_at_utc"] = utc_now()
    write_json(destination, report)
    print(report, flush=True)
    if not report["native_verification_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
