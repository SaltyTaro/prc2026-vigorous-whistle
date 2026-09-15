"""Launch one declared selected refit on the existing arcw0 worker."""
from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_surface_refit.training import model_code_receipt


def main():
    if socket.gethostname() != "arcw0" or ROOT != Path("/home/azureuser/prc-flight-2026"):
        raise RuntimeError("Use only the existing isolated arcw0 worker.")
    docs, runs = ROOT / "docs/surface_refit", ROOT / "runs/surface_refit"
    docs.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    if (docs / "launch.json").exists() or (runs / "refit.log").exists() or list(runs.glob("*_refit_*")):
        raise ValueError("Preserve the existing selected refit; do not duplicate its launch.")
    declaration = read_json(docs / "declaration.json")
    if declaration["model_code"] != model_code_receipt(ROOT):
        raise ValueError("The declared refit source changed.")
    active_tools = {"run_timeline.py", "run_arrivalctx.py", "run_surface_airports.py", "refit_surface.py"}
    for process in psutil.process_iter(["pid", "cmdline"]):
        if any(Path(part).name in active_tools for part in (process.info["cmdline"] or [])):
            raise ValueError("Another competition fit is still active.")
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = "16"
    started = utc_now()
    subprocess.run([sys.executable, "-u", "-B", "-m", "pytest", "-q", "tests/test_surface_refit.py", "--junitxml=docs/surface_refit/tests_azure.xml"], cwd=ROOT, check=True)
    with (runs / "refit.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, "-u", "-B", "tools/refit_surface.py"], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    report = {"created_at_utc": utc_now(), "pid": process.pid, "worker": socket.gethostname(), "threads": 16,
              "tests_started_at_utc": started, "tests_sha256": sha256_file(docs / "tests_azure.xml"),
              "declaration_sha256": sha256_file(docs / "declaration.json"), "model_code_sha256": declaration["model_code"]["sha256"],
              "free_disk_bytes_at_launch": shutil.disk_usage(ROOT).free, "new_native_models": 12,
              "audit_used": False, "submission_uploaded": False}
    write_json(docs / "launch.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
