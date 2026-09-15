"""Test and execute one assigned parallel job, preserving its status and log."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--start-at", required=True)
    args = parser.parse_args()
    job_path = ROOT / args.job
    job = json.loads(job_path.read_text())
    status_path = ROOT / "runs/nm_mask_status" / f"{job['job_id']}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if status_path.exists():
        raise ValueError("This job already has an execution record.")
    state = {"job_id": job["job_id"], "worker": job["worker"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "steps": [], "phase": "testing", "complete": False, "audit_used": False}
    def save():
        status_path.write_text(json.dumps(state, indent=2) + "\n")
    save()
    report = ROOT / "docs/nm_mask" / f"tests_{job['job_id']}.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    commands = [[sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 "tests/test_nm_mask_features.py", "tests/test_nm_mask_native.py", f"--junitxml={report}"],
                [sys.executable, "-B", "tools/run_nm_mask.py", "--job", args.job]]
    env = dict(os.environ, OMP_NUM_THREADS="16", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="16", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    for index, command in enumerate(commands):
        if index:
            state["phase"] = "waiting_for_common_start"
            save()
            start = datetime.fromisoformat(args.start_at)
            while (remaining := (start - datetime.now(timezone.utc)).total_seconds()) > 0:
                time.sleep(min(remaining, 1.))
            state["phase"] = "training"
            save()
        step = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command}
        result = subprocess.run(command, cwd=ROOT, env=env)
        step.update(returncode=result.returncode, finished_at_utc=datetime.now(timezone.utc).isoformat())
        state["steps"].append(step)
        if result.returncode:
            state["phase"] = "failed"
            save()
            raise SystemExit(result.returncode)
        save()
    state.update(phase="complete", complete=True, finished_at_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == "__main__":
    main()
