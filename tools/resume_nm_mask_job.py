"""Resume only a test-import failure, without changing or repeating a native fit."""
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
    job = json.loads((ROOT / args.job).read_text())
    old_path = ROOT / "runs/nm_mask_status" / f"{job['job_id']}.json"
    old = json.loads(old_path.read_text())
    if old["phase"] != "failed" or len(old["steps"]) != 1 or old["steps"][0]["returncode"] != 4:
        raise ValueError("Only the preserved pre-training pytest import failure may resume here.")
    if (ROOT / "runs/nm_mask" / job["job_id"]).exists():
        raise ValueError("A native model output already exists; do not repeat it.")
    status_path = old_path.with_name(job["job_id"] + "_resumed.json")
    if status_path.exists():
        raise ValueError("A resumed execution already exists.")
    state = {"job_id": job["job_id"], "worker": job["worker"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "steps": [], "phase": "testing", "complete": False, "audit_used": False,
             "preserved_failure": old_path.relative_to(ROOT).as_posix(), "fix": "Explicit PYTHONPATH points to the unchanged project src directory."}
    def save():
        status_path.write_text(json.dumps(state, indent=2) + "\n")
    save()
    report = ROOT / "docs/nm_mask" / f"tests_resumed_{job['job_id']}.xml"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="16", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="16", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    commands = [[sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 "tests/test_nm_mask_features.py", "tests/test_nm_mask_native.py", f"--junitxml={report}"],
                [sys.executable, "-B", "tools/run_nm_mask.py", "--job", args.job]]
    for index, command in enumerate(commands):
        if index:
            state["phase"] = "waiting_for_common_start"
            save()
            while (delay := (datetime.fromisoformat(args.start_at) - datetime.now(timezone.utc)).total_seconds()) > 0:
                time.sleep(min(delay, 1.))
            state["phase"] = "training"
            save()
        step = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "command": command}
        completed = subprocess.run(command, cwd=ROOT, env=env)
        step.update(returncode=completed.returncode, finished_at_utc=datetime.now(timezone.utc).isoformat())
        state["steps"].append(step)
        if completed.returncode:
            state["phase"] = "failed"
            save()
            raise SystemExit(completed.returncode)
        save()
    state.update(phase="complete", complete=True, finished_at_utc=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == "__main__":
    main()
