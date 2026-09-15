"""Execute the declared six-fit batch on the existing authorized Azure worker."""

from pathlib import Path
import json
import os
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from prc2026.artifacts import utc_now,write_json


def main():
    if socket.gethostname()!="arcw0" or ROOT!=Path("/home/azureuser/prc-flight-2026"):
        raise ValueError("This batch is limited to the authorized arcw0 workspace.")
    output=ROOT/"docs/rome_geometry/batch_completion.json"
    if output.exists(): raise ValueError("Preserve the existing batch receipt.")
    for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[key]="16"
    steps=[["-m","pytest","-q","tests/test_rome_geometry.py","tests/test_source_geometry_audit.py","--junitxml=docs/rome_geometry/tests_azure.xml"],
        ["tools/check_rome_geometry_features.py"],
        ["tools/run_rome_geometry.py","--fold","seasonal"],
        ["tools/run_rome_geometry.py","--fold","forward"],
        ["tools/select_rome_geometry.py"],["tools/verify_rome_geometry.py"]]
    records=[]
    for arguments in steps:
        row={"arguments":arguments,"started_at_utc":utc_now()}
        result=subprocess.run([sys.executable,"-u","-B",*arguments],cwd=ROOT,check=False)
        row.update(finished_at_utc=utc_now(),returncode=result.returncode)
        records.append(row)
        if result.returncode: break
    complete=len(records)==len(steps) and all(row["returncode"]==0 for row in records)
    write_json(output,{"created_at_utc":utc_now(),"declared_batch_complete":complete,"steps":records,
        "audit_used":False,"submission_uploaded":False,"new_native_models_declared":6})
    if not complete: raise SystemExit(records[-1]["returncode"])


if __name__=="__main__": main()
