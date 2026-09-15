"""Launch the bounded batch inside the already authorized arcw0 workspace."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess

root = Path(__file__).resolve().parents[1]
if root != Path("/home/azureuser/prc-flight-2026") or socket.gethostname() != "arcw0":
    raise ValueError("The training batch is restricted to the allocated arcw0 workspace.")
output = root/"runs/phase5r"
output.mkdir(parents=True, exist_ok=True)
receipt_path = output/"launch.json"
if receipt_path.exists():
    raise ValueError("A launch receipt already exists; inspect that process before another launch.")
with (output/"batch.log").open("xb") as log:
    process = subprocess.Popen(["bash", "infra/azure/phase5r.sh"], cwd=root, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "pid": process.pid,
           "host": socket.gethostname(), "workspace": str(root), "log": "runs/phase5r/batch.log"}
receipt_path.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps(receipt))
