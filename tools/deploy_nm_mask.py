"""Package, install and launch eight isolated Azure model/fold jobs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/azureuser/prc-flight-2026"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def ssh(worker, code, timeout=90):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", worker,
                             "/usr/bin/python3", "-B", "-"], input=code,
                            text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{worker}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def package(item):
    job_path = ROOT / item["path"]
    if sha(job_path) != item["sha256"]:
        raise ValueError("A bound job definition changed.")
    job = read(job_path)
    names = set(job["files"]) | {item["path"]}
    for directory in (ROOT / "src").glob("prc2026*"):
        names.update(p.relative_to(ROOT).as_posix() for p in directory.rglob("*.py") if "__pycache__" not in p.parts)
    names.update(["configs/nm_mask_v1.json", "configs/features_nm_mask_v1.json", "configs/validation_v1.json",
                  "requirements-phase1.txt", "requirements-phase2.txt", "requirements-nm-mask.txt",
                  "infra/azure/parallel_authorization_20260915.json", "tools/run_nm_mask.py", "tools/batch_nm_mask_job.py",
                  "tests/test_nm_mask_features.py", "tests/test_nm_mask_native.py"])
    if (ROOT / "tests/conftest.py").exists():
        names.add("tests/conftest.py")
    files = {name: sha(ROOT / name) for name in sorted(names)}
    receipt_path = ROOT / "docs/nm_mask/packages" / f"{item['job_id']}.json"
    archive_path = ROOT / "runs/nm_mask_inputs" / f"{item['job_id']}.tar.gz"
    if receipt_path.exists() or archive_path.exists():
        raise ValueError("Input package already exists; preserve the original.")
    manifest_path = ROOT / "docs/nm_mask/package_manifests" / f"{item['job_id']}.json"
    write_new(manifest_path, {"job": item, "files": files, "contains_credentials": False,
                              "contains_december_labels": False})
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "x:gz", compresslevel=1) as archive:
        for name in files:
            path = ROOT / name
            if not path.resolve().is_relative_to(ROOT) or path.is_symlink() or ".env" in path.parts:
                raise ValueError("Unapproved package path.")
            archive.add(path, arcname=name, recursive=False)
        archive.add(manifest_path, arcname=manifest_path.relative_to(ROOT).as_posix(), recursive=False)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "job": item,
              "archive": archive_path.relative_to(ROOT).as_posix(), "sha256": sha(archive_path),
              "size_bytes": archive_path.stat().st_size, "manifest": manifest_path.relative_to(ROOT).as_posix(),
              "unpacked_bytes": sum((ROOT / n).stat().st_size for n in files), "file_count": len(files) + 1}
    write_new(receipt_path, report)
    print(json.dumps({"packaged": item["job_id"], "bytes": report["size_bytes"]}), flush=True)
    return report


def install(report):
    item = report["job"]
    worker = item["worker"]
    incoming = f"{REMOTE}/incoming/nm_mask/{item['job_id']}-{report['sha256'][:16]}.tar.gz"
    preflight = f'''
from pathlib import Path
import json,platform,shutil
root=Path({REMOTE!r});path=Path({incoming!r})
assert platform.node()=={worker!r}
assert path.resolve().is_relative_to(root.resolve())
assert not path.exists(),'An input transfer already exists.'
assert shutil.disk_usage(root).free>{report['size_bytes']+report['unpacked_bytes']+400000000},'Insufficient project disk space.'
path.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({{'ready':True,'free_bytes':shutil.disk_usage(root).free}}))
'''
    state = ssh(worker, preflight)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                    str(ROOT / report["archive"]), f"{worker}:{incoming}"], check=True, timeout=600)
    code = f'''
from pathlib import Path,PurePosixPath
import json,hashlib,tarfile,platform,shutil
root=Path({REMOTE!r});path=Path({incoming!r})
assert platform.node()=={worker!r}
with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={report['sha256']!r}
written=matched=0
with tarfile.open(path,'r:gz') as archive:
 members=archive.getmembers();assert len(members)=={report['file_count']}
 assert len({{m.name for m in members}})==len(members)
 manifest=json.load(archive.extractfile({report['manifest']!r}))
 expected=set(manifest['files'])|{{{report['manifest']!r}}}
 assert {{m.name for m in members}}==expected
 for member in members:
  rel=PurePosixPath(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and chr(92) not in member.name
  target=root/member.name;assert target.resolve().is_relative_to(root.resolve())
  payload=archive.extractfile(member).read()
  if member.name in manifest['files']:assert hashlib.sha256(payload).hexdigest()==manifest['files'][member.name]
  if target.exists():
   assert target.read_bytes()==payload,'Existing differing project file: '+member.name
   matched+=1
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write(payload)
   written+=1
print(json.dumps({{'worker':platform.node(),'written_files':written,'unchanged_files':matched,'disk_free_bytes':shutil.disk_usage(root).free,'input_sha256':{report['sha256']!r}}}))
'''
    installed = ssh(worker, code, timeout=180)
    receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "job": item,
               "preflight": state, "installed": installed, "remote_archive": incoming}
    write_new(ROOT / "docs/nm_mask" / f"deployment_{worker}.json", receipt)
    print(json.dumps({"installed": worker, "job": item["job_id"]}), flush=True)
    return receipt


def launch(item, start_at):
    worker = item["worker"]
    code = f'''
from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,platform,hashlib,os
root=Path({REMOTE!r});job=root/{item['path']!r}
assert platform.node()=={worker!r}
assert hashlib.sha256(job.read_bytes()).hexdigest()=={item['sha256']!r}
receipt=root/'docs/nm_mask'/('launch_'+{item['job_id']!r}+'.json')
assert not receipt.exists()
assert not (root/'runs/nm_mask'/{item['job_id']!r}).exists()
log=root/'runs/nm_mask_logs'/({item['job_id']!r}+'.log');log.parent.mkdir(parents=True,exist_ok=True)
command=[str(root/'.venv/bin/python'),'-B','tools/batch_nm_mask_job.py','--job',{item['path']!r},'--start-at',{start_at!r}]
env=dict(os.environ,OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='16',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
with log.open('xb') as output:process=subprocess.Popen(command,cwd=root,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,env=env)
data={{'created_at_utc':datetime.now(timezone.utc).isoformat(),'worker':platform.node(),'job_id':{item['job_id']!r},'pid':process.pid,'start_at_utc':{start_at!r},'log':str(log.relative_to(root)),'job_sha256':{item['sha256']!r}}}
with receipt.open('x') as f:json.dump(data,f,indent=2)
print(json.dumps(data))
'''
    result = ssh(worker, code)
    write_new(ROOT / "docs/nm_mask" / f"launch_{worker}.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["install", "launch"])
    args = parser.parse_args()
    jobs = read(ROOT / "docs/nm_mask/jobs_bound.json")["jobs"]
    if {j["worker"] for j in jobs} != {f"arcw{i}" for i in range(8)}:
        raise ValueError("The eight-worker assignment changed.")
    if args.action == "install":
        reports = [package(item) for item in jobs]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(install, report) for report in reports]
            results = [future.result() for future in as_completed(futures)]
        write_new(ROOT / "docs/nm_mask/deployment.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "workers": results})
    else:
        feature_report = read(ROOT / "docs/nm_mask/local_feature_verification.json")
        if not feature_report["passed"] or feature_report["audit_used"]:
            raise ValueError("Feature verification must pass before fitting.")
        cases = ET.parse(ROOT / "docs/nm_mask/tests_local.xml").findall(".//testcase")
        if len(cases) != 34 or any(c.find(tag) is not None for c in cases for tag in ["failure", "error", "skipped"]):
            raise ValueError("All 34 local checks must pass before launching.")
        if len(read(ROOT / "docs/nm_mask/deployment.json")["workers"]) != 8:
            raise ValueError("Every worker must have verified inputs before launching.")
        start_at = (datetime.now(timezone.utc) + timedelta(seconds=25)).isoformat()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(launch, item, start_at) for item in jobs]
            results = [future.result() for future in as_completed(futures)]
        write_new(ROOT / "docs/nm_mask/launch.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "workers": results})


if __name__ == "__main__":
    main()
