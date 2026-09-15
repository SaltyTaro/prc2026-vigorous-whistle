"""Verify the frozen eight-worker results on arcw4, preserving every input."""
from pathlib import Path
import json
import subprocess
import tarfile

from deploy_nm_mask import ROOT, REMOTE, read, sha, ssh, write_new


WORKER = "arcw4"


def main():
    baseline = read(ROOT / "docs/nm_mask/package_manifests/seasonal_lgb_augmented.json")["files"]
    names = set(baseline)
    names.update(["tools/select_nm_mask.py", "tools/verify_nm_mask.py",
                  "tests/test_nm_mask_verification.py", "docs/nm_mask/jobs_bound.json",
                  "docs/nm_mask/development_results.json", "data/nm_mask/latest.json"])
    for item in read(ROOT / "docs/nm_mask/jobs_bound.json")["jobs"]:
        names.add(item["path"])
        names.update(read(ROOT / item["path"])["files"])
    names.update(p.relative_to(ROOT).as_posix()
                 for p in (ROOT / "runs/azure_parallel_nm_mask").rglob("*") if p.is_file())
    bundle = read(ROOT / "data/nm_mask/latest.json")["bundle"]
    names.add(bundle + "/manifest.json")
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / bundle / "routing").glob("*.parquet"))
    required = {name: sha(ROOT / name) for name in sorted(names)}
    for name, digest in baseline.items():
        if required[name] != digest:
            raise ValueError("An installed source or input changed: " + name)
    delta = {name: digest for name, digest in required.items() if name not in baseline}
    manifest_name = "docs/nm_mask/verification_package_manifest.json"
    write_new(ROOT / manifest_name, {"worker": WORKER, "files": delta, "required_files": required,
        "feature_bundle_projection": "Manifest and original routing only; model job inputs are complete.",
        "contains_credentials": False, "contains_december_labels": False})
    archive_name = "runs/nm_mask_inputs/verification_arcw4.tar.gz"
    archive_path = ROOT / archive_name
    with tarfile.open(archive_path, "x:gz", compresslevel=1) as archive:
        for name in [*delta, manifest_name]:
            path = ROOT / name
            if not path.resolve().is_relative_to(ROOT) or path.is_symlink() or ".env" in path.parts:
                raise ValueError("Unapproved package path.")
            archive.add(path, arcname=name, recursive=False)
    archive_hash = sha(archive_path)
    report = {"worker": WORKER, "archive": archive_name, "sha256": archive_hash,
        "manifest": manifest_name, "file_count": len(delta) + 1, "required_file_count": len(required),
        "size_bytes": archive_path.stat().st_size,
        "unpacked_bytes": sum((ROOT / name).stat().st_size for name in delta)}
    write_new(ROOT / "docs/nm_mask/verification_package.json", report)
    incoming = REMOTE + "/incoming/nm_mask/verification-" + archive_hash[:16] + ".tar.gz"
    ready = ssh(WORKER, f'''
from pathlib import Path
import json,platform,shutil
root=Path({REMOTE!r});path=Path({incoming!r})
assert platform.node()=={WORKER!r}
assert path.resolve().is_relative_to(root.resolve()) and not path.exists()
assert shutil.disk_usage(root).free>{report['size_bytes'] + report['unpacked_bytes'] + 1000000000}
path.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({{'ready':True,'disk_free_bytes':shutil.disk_usage(root).free}}))
''')
    print(json.dumps({"stage": "verification_package_ready", **report}), flush=True)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(archive_path), WORKER + ":" + incoming],
                   check=True, timeout=600)
    installed = ssh(WORKER, f'''
from pathlib import Path,PurePosixPath
import json,hashlib,tarfile,platform,shutil
root=Path({REMOTE!r});path=Path({incoming!r})
assert platform.node()=={WORKER!r}
with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={archive_hash!r}
written=matched=0
with tarfile.open(path,'r:gz') as archive:
 members=archive.getmembers();assert len(members)=={len(delta)+1}
 assert len({{m.name for m in members}})==len(members)
 manifest=json.load(archive.extractfile({manifest_name!r}))
 assert {{m.name for m in members}}==set(manifest['files'])|{{{manifest_name!r}}}
 for member in members:
  rel=PurePosixPath(member.name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and chr(92) not in member.name
  target=root/member.name;assert target.resolve().is_relative_to(root.resolve())
  payload=archive.extractfile(member).read()
  if member.name in manifest['files']:assert hashlib.sha256(payload).hexdigest()==manifest['files'][member.name]
  if target.exists():
   assert target.read_bytes()==payload,'Existing differing file: '+member.name
   matched+=1
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write(payload)
   written+=1
for name,digest in manifest['required_files'].items():
 with (root/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,'Required input differs: '+name
print(json.dumps({{'worker':platform.node(),'written_files':written,'unchanged_files':matched,
 'required_files_checked':len(manifest['required_files']),'disk_free_bytes':shutil.disk_usage(root).free}}))
''', timeout=180)
    write_new(ROOT / "docs/nm_mask/verification_deployment.json", {"preflight": ready, "installed": installed,
        "remote_archive": incoming, "archive_sha256": archive_hash})
    print(json.dumps({"stage": "verification_installed", **installed}), flush=True)
    checked = ssh(WORKER, f'''
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,platform,hashlib
root=Path({REMOTE!r});assert platform.node()=={WORKER!r}
log=root/'runs/nm_mask_verification.log';output=root/'docs/nm_mask/verification.json'
test=root/'docs/nm_mask/tests_verifier_azure.xml';receipt=root/'docs/nm_mask/verification_execution.json'
assert not any(p.exists() for p in [log,output,test,receipt])
env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
commands=[['-m','pytest','-q','-p','no:cacheprovider','tests/test_nm_mask_verification.py','--junitxml='+str(test)],
 ['tools/verify_nm_mask.py','--output','docs/nm_mask/verification.json']]
steps=[]
with log.open('xb') as handle:
 for arguments in commands:
  command=[str(root/'.venv/bin/python'),'-B',*arguments]
  step={{'command':command,'started_at_utc':datetime.now(timezone.utc).isoformat()}}
  result=subprocess.run(command,cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT)
  step.update(returncode=result.returncode,finished_at_utc=datetime.now(timezone.utc).isoformat());steps.append(step)
  if result.returncode:break
record={{'worker':platform.node(),'steps':steps,'passed':len(steps)==2 and all(s['returncode']==0 for s in steps),
 'audit_used':False,'artifacts':{{}}}}
for p in [log,test,output]:
 if p.exists():
  with p.open('rb') as f:record['artifacts'][p.relative_to(root).as_posix()]=hashlib.file_digest(f,'sha256').hexdigest()
with receipt.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
''', timeout=900)
    write_new(ROOT / "docs/nm_mask/verification_execution.json", checked)
    for name, digest in checked["artifacts"].items():
        target = ROOT / name
        if target.exists():
            raise ValueError("Preserve the existing local verification artifact: " + name)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", WORKER + ":" + REMOTE + "/" + name, str(target)],
                       check=True, timeout=120)
        if sha(target) != digest:
            raise ValueError("Verification mirror differs: " + name)
    print(json.dumps({"stage": "verification_complete", "passed": checked["passed"]}), flush=True)
    if not checked["passed"]:
        raise RuntimeError("Azure verification failed; preserved execution and logs.")


if __name__ == "__main__":
    main()
