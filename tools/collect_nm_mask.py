"""Mirror completed worker artifacts with archive/member checks; never overwrite."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def collect(item):
    receipt_path = ROOT / "docs/nm_mask/mirrors" / f"{item['worker']}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["job"] != item or sha(ROOT / receipt["archive"]) != receipt["sha256"]:
            raise ValueError("Existing mirror identity changed.")
        return {"worker": item["worker"], "already_mirrored": True}
    code = f'''
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,platform
root=Path('/home/azureuser/prc-flight-2026');job={item['job_id']!r};assert platform.node()=={item['worker']!r}
status=root/'runs/nm_mask_status'/(job+'_resumed.json')
if not status.exists():status=root/'runs/nm_mask_status'/(job+'.json')
if not status.exists() or not json.loads(status.read_text()).get('complete'):
 print(json.dumps({{'pending':True,'worker':platform.node()}}));raise SystemExit(0)
run=root/'runs/nm_mask'/job;assert (run/'run.json').exists()
paths=[p for p in run.iterdir() if p.is_file()]
for folder,patterns in [('runs/nm_mask_status',[job+'.json',job+'_resumed.json']),('runs/nm_mask_logs',[job+'.log',job+'_resumed.log']),('docs/nm_mask',['launch_'+job+'.json','launch_resumed_'+job+'.json','tests_'+job+'.xml','tests_resumed_'+job+'.xml'])]:
 paths.extend(root/folder/name for name in patterns if (root/folder/name).exists())
files={{}}
for p in sorted(paths):
 with p.open('rb') as f:files[p.relative_to(root).as_posix()]=hashlib.file_digest(f,'sha256').hexdigest()
archive=root/'incoming/nm_mask'/('results_'+job+'.tar.gz');receipt=archive.with_suffix(archive.suffix+'.json')
if archive.exists():
 old=json.loads(receipt.read_text());assert old['files']==files
else:
 with tarfile.open(archive,'x:gz',compresslevel=1) as tf:
  for name in files:tf.add(root/name,arcname=name,recursive=False)
 with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 old={{'created_at_utc':datetime.now(timezone.utc).isoformat(),'worker':platform.node(),'path':str(archive),'sha256':digest,'size_bytes':archive.stat().st_size,'files':files}}
 with receipt.open('x') as f:json.dump(old,f,indent=2,sort_keys=True)
print(json.dumps(old))
'''
    compile(code, "collect_remote", "exec")
    remote = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", item["worker"],
                             "/usr/bin/python3", "-B", "-"], input=code, text=True,
                            capture_output=True, timeout=120, check=True)
    report = json.loads(remote.stdout)
    if report.get("pending"):
        return report
    archive_path = ROOT / "runs/nm_mask_results" / f"{item['job_id']}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise ValueError("A result archive exists without a mirror receipt; preserve it for review.")
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                    f"{item['worker']}:{report['path']}", str(archive_path)], check=True, timeout=240)
    if sha(archive_path) != report["sha256"] or archive_path.stat().st_size != report["size_bytes"]:
        raise ValueError("Transferred archive digest differs.")
    destinations = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(report["files"]) or len({m.name for m in members}) != len(members):
            raise ValueError("Result archive identities differ.")
        for member in members:
            relative = PurePosixPath(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts or "\\" in member.name or member.name not in report["files"]:
                raise ValueError("Unsafe archive member.")
            prefix = f"runs/nm_mask/{item['job_id']}/"
            if member.name.startswith(prefix):
                destination = ROOT / "runs/azure_parallel_nm_mask" / item["job_id"] / member.name[len(prefix):]
            else:
                destination = ROOT / "docs/nm_mask/worker_snapshots" / item["worker"] / member.name
            if not destination.resolve().is_relative_to(ROOT):
                raise ValueError("Result restore escaped the project.")
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != report["files"][member.name]:
                raise ValueError("Result member digest differs.")
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise ValueError("A differing result file already exists.")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(payload)
            destinations[member.name] = destination.relative_to(ROOT).as_posix()
    receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "job": item,
               "archive": archive_path.relative_to(ROOT).as_posix(), "sha256": report["sha256"],
               "size_bytes": report["size_bytes"], "files": report["files"], "destinations": destinations,
               "remote_archive": report["path"], "every_member_readback_matches": True}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"worker": item["worker"], "mirrored": True, "files": len(destinations), "bytes": report["size_bytes"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    jobs = json.loads((ROOT / "docs/nm_mask/jobs_bound.json").read_text())["jobs"]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(collect, job) for job in jobs]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    if args.require_all and any(r.get("pending") for r in results):
        raise SystemExit("Some declared jobs are still running; do not select a partial batch.")


if __name__ == "__main__":
    main()
