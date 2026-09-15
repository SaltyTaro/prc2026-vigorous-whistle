"""Launch the declared ladder CPU batch only in the existing arcw0 workspace."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import subprocess

root = Path(__file__).resolve().parents[1]
if root != Path("/home/azureuser/prc-flight-2026") or socket.gethostname() != "arcw0":
    raise ValueError("This launcher is restricted to the authorized arcw0 workspace.")
import sys
sys.path.insert(0, str(root/"src"))
from prc2026.artifacts import read_json
from prc2026_ladder_scale.training import promotion_reference, model_code_receipt
promotion_reference(root, read_json(root/"configs/ladder_scale_v1.json"))
if model_code_receipt(root)["sha256"] != read_json(root/"docs/ladder_scale/declaration.json")["model_code_sha256"]:
    raise ValueError("The declared target-scale model code changed before launch.")
import psutil
prior_launch = read_json(root/"runs/ladder/launch.json")
try:
    prior_process = psutil.Process(prior_launch["pid"])
    if prior_process.is_running() and prior_process.status() != psutil.STATUS_ZOMBIE:
        raise ValueError("The preceding batch must finish before using its worker.")
except psutil.NoSuchProcess:
    pass
output = root/"runs/ladder_scale"
output.mkdir(parents=True, exist_ok=True)
if (output/"launch.json").exists() or (output/"batch.log").exists():
    raise ValueError("An existing ladder launch must be inspected rather than duplicated.")
if shutil.disk_usage(root).free < 2_000_000_000:
    raise ValueError("At least 2 GB free is required before the declared batch.")
with (output/"batch.log").open("xb") as log:
    process = subprocess.Popen(["bash", "infra/azure/ladder_scale.sh"], cwd=root, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "pid": process.pid,
           "host": socket.gethostname(), "workspace": str(root), "log": "runs/ladder_scale/batch.log",
           "new_workers": 0, "automatic_submission": False}
(output/"launch.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps(receipt))


