"""Read-only timeline batch status and completed native metrics."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import psutil
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    active = []
    names = {"batch_timeline.py", "resume_timeline.py", "prepare_timeline.py", "run_timeline.py", "select_timeline.py", "verify_timeline.py"}
    for process in psutil.process_iter(["pid", "cmdline", "cwd", "memory_info"]):
        try:
            if process.info["cwd"] == str(ROOT) and any(Path(part).name in names for part in process.info["cmdline"] or []):
                active.append({"pid": process.pid, "command": process.info["cmdline"], "rss_bytes": process.info["memory_info"].rss})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    tests = ROOT/"docs/timeline/tests_azure.xml"
    if (ROOT/"docs/timeline/import_fix/tests_azure.xml").exists():
        tests = ROOT/"docs/timeline/import_fix/tests_azure.xml"
    count = None
    if tests.exists():
        count = len(ET.parse(tests).findall(".//testcase"))
    runs = []
    for path in sorted((ROOT/"runs/timeline").glob("*/run.json")):
        report = json.loads(path.read_text())
        metrics = json.loads(path.with_name("metrics.json").read_text())
        runs.append({"variant": report["variant"], "fold": report["fold"], "iterations": report["best_iterations"],
                     "complete_global_only_rmse": metrics["overall"]["rmse_seconds"], "wall_seconds": report["resources"]["wall_seconds"]})
    pointer = ROOT/"data/timeline/latest.json"
    selected = ROOT/"runs/timeline/selection.json"
    print(json.dumps({"tests": count, "active": active, "completed_native_runs": runs,
                      "bundle": json.loads(pointer.read_text()) if pointer.exists() else None,
                      "selection_complete": selected.exists(), "free_disk_bytes": shutil.disk_usage(ROOT).free}, indent=2))


if __name__ == "__main__":
    main()
