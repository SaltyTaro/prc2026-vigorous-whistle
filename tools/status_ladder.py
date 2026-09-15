"""Read the current ladder job and its saved full-development results."""
import json
from pathlib import Path
import sys
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, utc_now

root = Path(__file__).resolve().parents[1]
output = root/"runs/ladder"
launch = read_json(output/"launch.json")
pid = launch["pid"]
active = False
processes = []
try:
    process = psutil.Process(pid)
    active = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    if active:
        for item in [process, *process.children(recursive=True)]:
            try:
                processes.append({"pid": item.pid, "name": item.name(), "rss_bytes": item.memory_info().rss})
            except psutil.Error:
                pass
except psutil.Error:
    pass
runs = []
for path in sorted(output.glob("*/run.json")):
    report = read_json(path)
    if report["stage"] == "development":
        metric = read_json(path.parent/"metrics.json")["overall"]
        runs.append({"run_id": path.parent.name, "variant": report["variant"], "fold": report["fold"],
                     "rmse_seconds": metric["rmse_seconds"], "rows": metric["rows"],
                     "iterations": report["best_iterations"], "wall_seconds": report["resources"]["wall_seconds"]})
lines = (output/"batch.log").read_text(errors="replace").splitlines()
print(json.dumps({"checked_at_utc": utc_now(), "launch": launch, "active": active,
                  "processes": processes, "completed_development_runs": runs, "completed_count": len(runs),
                  "disk_free_bytes": psutil.disk_usage(root).free, "tail": lines[-5:]}, indent=2))
