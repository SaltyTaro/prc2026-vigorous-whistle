"""Read-only status of the eight explicitly assigned Azure jobs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def inspect(item):
    code = f'''
from pathlib import Path
import json,psutil,platform
root=Path('/home/azureuser/prc-flight-2026');job={item['job_id']!r}
assert platform.node()=={item['worker']!r}
status=root/'runs/nm_mask_status'/(job+'.json')
launch=root/'docs/nm_mask'/('launch_'+job+'.json')
resumed=root/'runs/nm_mask_status'/(job+'_resumed.json')
if resumed.exists():
 status=resumed
 launch=root/'docs/nm_mask'/('launch_resumed_'+job+'.json')
result={{'worker':platform.node(),'job_id':job,'status':json.loads(status.read_text()) if status.exists() else None}}
run=root/'runs/nm_mask'/job/'run.json'
if run.exists():
 receipt=json.loads(run.read_text());result['result']={{k:receipt[k] for k in ['iterations','complete_rmse_seconds','resources','native_reload_maximum_difference_seconds']}}
if launch.exists():
 launch_data=json.loads(launch.read_text());pid=launch_data['pid'];processes=[]
 try:
  parent=psutil.Process(pid)
  if any(name in parent.cmdline() for name in ['tools/batch_nm_mask_job.py','tools/resume_nm_mask_job.py']):
   for p in parent.children(recursive=True):
    try:processes.append({{'pid':p.pid,'rss_bytes':p.memory_info().rss,'cpu_seconds':sum(p.cpu_times()[:2])}})
    except psutil.Error:pass
 except psutil.Error:pass
 result['child_processes']=processes
 log=root/launch_data['log']
 if log.exists():result['log_tail']=log.read_text(errors='replace').splitlines()[-16:]
print(json.dumps(result))
'''
    process = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", item["worker"],
        "/home/azureuser/prc-flight-2026/.venv/bin/python", "-B", "-"], input=code,
        text=True, capture_output=True, timeout=40)
    if process.returncode:
        return {"worker": item["worker"], "job_id": item["job_id"], "error": process.stderr.strip()}
    return json.loads(process.stdout)


def main():
    jobs = json.loads((ROOT / "docs/nm_mask/jobs_bound.json").read_text())["jobs"]
    with ThreadPoolExecutor(max_workers=8) as executor:
        workers = list(executor.map(inspect, jobs))
    timestamp = datetime.now(timezone.utc)
    report = {"checked_at_utc": timestamp.isoformat(), "read_only": True, "workers": workers}
    destination = ROOT / "docs/nm_mask/status" / (timestamp.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for worker in workers:
        print(json.dumps({"worker": worker["worker"], "job": worker["job_id"],
            "phase": worker.get("status", {}).get("phase"), "result": worker.get("result"),
            "active_children": len(worker.get("child_processes", [])), "error": worker.get("error"),
            "log_tail": worker.get("log_tail") if worker.get("status", {}).get("phase") == "failed" else None}), flush=True)


if __name__ == "__main__":
    main()
