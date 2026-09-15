"""Launch the declared cat_long refit once on the existing arcw0 worker, after its tests pass."""
from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026_v10_refit.training import preflight, refit_code_receipt


def main():
    if socket.gethostname() != "arcw0" or ROOT != Path("/home/azureuser/prc-flight-2026"):
        raise RuntimeError("Use only the existing isolated arcw0 worker.")
    docs, runs = ROOT / "docs/v10", ROOT / "runs/v10"
    docs.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    if (docs / "refit_launch.json").exists() or (runs / "refit.log").exists() or list(runs.glob("*_refit_*")):
        raise ValueError("Preserve the existing refit; do not duplicate its launch.")
    config_path, selection_path = ROOT / "configs/v10_v2.json", runs / "selection_v2.json"
    context = preflight(ROOT, config_path, selection_path)
    active_tools = {"run_timeline.py", "run_arrivalctx.py", "run_surface_airports.py", "refit_surface.py", "run_ladder.py", "refit_ladder.py", "refit_v10_cat_long.py"}
    for process in psutil.process_iter(["pid", "cmdline"]):
        if any(Path(part).name in active_tools for part in (process.info["cmdline"] or [])):
            raise ValueError("Another competition fit is still active.")
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = str(context["spec"]["threads"])
    started = utc_now()
    subprocess.run([sys.executable, "-u", "-B", "-m", "pytest", "-q", "tests/test_v10.py", "--junitxml=docs/v10/tests_azure.xml"], cwd=ROOT, check=True)
    with (runs / "refit.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, "-u", "-B", "tools/refit_v10_cat_long.py"], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    report = {"created_at_utc": utc_now(), "pid": process.pid, "process_created": psutil.Process(process.pid).create_time(),
              "worker": socket.gethostname(), "threads": context["spec"]["threads"], "iterations": context["spec"]["iterations"],
              "tests_started_at_utc": started, "tests_sha256": sha256_file(docs / "tests_azure.xml"),
              "config_sha256": object_hash(context["config"]), "selection_sha256": sha256_file(selection_path),
              "selected_candidate": context["selection"]["selected_candidate"], "model_code_sha256": refit_code_receipt(ROOT)["sha256"],
              "free_disk_bytes_at_launch": shutil.disk_usage(ROOT).free, "new_native_models": 1, "audit_used": False, "submission_uploaded": False}
    write_json(docs / "refit_launch.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
