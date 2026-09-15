"""Launch one isolated Rome comparison after checking its frozen declaration."""

from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from prc2026.artifacts import read_json,sha256_file,utc_now,write_json
from prc2026_rome_geometry.common import configurations,model_code_receipt
from prc2026_rome_geometry.bundle import feature_code_receipt


def main():
    if socket.gethostname()!="arcw0" or ROOT!=Path("/home/azureuser/prc-flight-2026"):
        raise ValueError("Use only the authorized arcw0 workspace.")
    _,config=configurations(ROOT)
    docs,runs=ROOT/"docs/rome_geometry",ROOT/"runs/rome_geometry"
    if (docs/"launch.json").exists() or (runs/"batch.log").exists() or list(runs.glob("*/run.json")):
        raise ValueError("Do not duplicate the geometry launch.")
    declaration=read_json(docs/"declaration_after_preparation_fix.json")
    if declaration["model_code"]!=model_code_receipt(ROOT) or declaration["feature_code"]!=feature_code_receipt(ROOT):
        raise ValueError("The frozen launch source differs from the declaration.")
    for process in psutil.process_iter(["pid","cmdline"]):
        args=process.info["cmdline"] or []
        if any(Path(part).name.startswith(("run_","refit_","batch_")) and Path(part).suffix==".py" for part in args if part.startswith("tools/")):
            raise ValueError("Another competition training process is active.")
    if shutil.disk_usage(ROOT).free<config["execution"]["minimum_free_disk_before_fit_bytes"]:
        raise ValueError("The declared free-disk margin is unavailable.")
    for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[key]="16"
    runs.mkdir(parents=True,exist_ok=True)
    with (runs/"batch.log").open("xb") as log:
        process=subprocess.Popen([sys.executable,"-u","-B","tools/batch_rome_geometry.py"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    report={"created_at_utc":utc_now(),"pid":process.pid,"worker":socket.gethostname(),"threads":16,
        "free_disk_bytes":shutil.disk_usage(ROOT).free,"declaration_sha256":sha256_file(docs/"declaration_after_preparation_fix.json"),
        "model_code_sha256":declaration["model_code"]["sha256"],"new_native_models":6,"audit_used":False,"submission_uploaded":False}
    write_json(docs/"launch.json",report)
    print(report,flush=True)


if __name__=="__main__": main()
