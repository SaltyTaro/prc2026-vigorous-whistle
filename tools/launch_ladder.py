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
output = root/"runs/ladder"
output.mkdir(parents=True, exist_ok=True)
if (output/"launch.json").exists() or (output/"batch.log").exists():
    raise ValueError("An existing ladder launch must be inspected rather than duplicated.")
if shutil.disk_usage(root).free < 2_000_000_000:
    raise ValueError("At least 2 GB free is required before the declared batch.")
with (output/"batch.log").open("xb") as log:
    process = subprocess.Popen(["bash", "infra/azure/ladder.sh"], cwd=root, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "pid": process.pid,
           "host": socket.gethostname(), "workspace": str(root), "log": "runs/ladder/batch.log",
           "new_workers": 0, "automatic_submission": False}
(output/"launch.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps(receipt))
