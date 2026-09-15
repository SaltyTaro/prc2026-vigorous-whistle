"""Launch only the declared CPU batch in the existing arcw0 project workspace."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess

root = Path(__file__).resolve().parents[1]
if root != Path("/home/azureuser/prc-flight-2026") or socket.gethostname() != "arcw0":
    raise ValueError("The nmclock batch is restricted to the authorized arcw0 workspace.")
output = root/"runs/nmclock"
output.mkdir(parents=True, exist_ok=True)
if (output/"launch.json").exists():
    raise ValueError("A nmclock launch receipt already exists; inspect the existing batch first.")
with (output/"batch.log").open("xb") as log:
    process = subprocess.Popen(["bash", "infra/azure/nmclock.sh"], cwd=root, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "pid": process.pid,
           "host": socket.gethostname(), "workspace": str(root), "log": "runs/nmclock/batch.log"}
(output/"launch.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps(receipt))
