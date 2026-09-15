"""Copy reviewed new files to arcw0 without overwriting any existing artifact."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from prc2026.artifacts import read_json,sha256_file,utc_now,write_json
from prc2026_rome_geometry.common import configurations,model_code_receipt
from prc2026_rome_geometry.bundle import feature_code_receipt

REMOTE="/home/azureuser/prc-flight-2026"


def remote_python(source):
    result=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15","arcw0",REMOTE+"/.venv/bin/python","-u","-B","-"],
        input=source,text=True,encoding="utf-8",capture_output=True,check=False)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return json.loads(result.stdout)


def main():
    config,_=configurations(ROOT)
    feature_report=read_json(ROOT/"docs/rome_geometry/local_feature_verification.json")
    if not feature_report["passed"]: raise ValueError("Local feature verification must pass before deployment.")
    declaration=read_json(ROOT/"docs/rome_geometry/declaration_after_preparation_fix.json")
    if declaration["model_code"]!=model_code_receipt(ROOT) or declaration["feature_code"]!=feature_code_receipt(ROOT):
        raise ValueError("Declared code changed before deployment.")
    package=ROOT/"runs/rome-geometry-source-inputs.tar.gz"
    receipt_path=ROOT/"docs/rome_geometry/initial_package.json"
    if package.exists() or receipt_path.exists(): raise ValueError("Preserve the reviewed input package.")
    members=list((ROOT/"src/prc2026_rome_geometry").glob("*.py"))+list((ROOT/"src/prc2026_source_geometry").glob("*.py"))
    tool_names=["prepare_rome_geometry.py","run_rome_geometry.py","check_rome_geometry_features.py","select_rome_geometry.py",
        "verify_rome_geometry.py","batch_rome_geometry.py","launch_rome_geometry.py","deploy_rome_geometry.py"]
    members += [ROOT/"tools"/name for name in tool_names]
    members += [ROOT/"configs/features_rome_geometry_v1.json",ROOT/"configs/rome_geometry_v1.json",
        ROOT/"requirements-rome-geometry.txt",ROOT/"tests/test_rome_geometry.py",ROOT/"tests/test_source_geometry_audit.py"]
    members += list((ROOT/"configs").glob("source_geometry*_v1.json"))
    members += [ROOT/name for name in config["input_receipts_sha256"]]
    members += list((ROOT/"docs/rome_geometry").glob("*.json"))+list((ROOT/"docs/rome_geometry").glob("*.xml"))
    members += [p for p in (ROOT/"docs/rome_geometry/preparation_corrections").rglob("*") if p.is_file()]
    members += [ROOT/"docs/source_geometry/EXTERNAL_DATA_LICENSES.md"]
    audit=read_json(ROOT/"docs/source_geometry/geometry_endpoint_review.json")
    for stamp in sorted(set(config["snapshot_assignment"].values())):
        source=audit["source_receipts"]["LIRF/"+stamp]
        members.append(ROOT/source["path"])
        members.append(ROOT/"docs/source_geometry/sources"/(Path(source["path"]).name+".receipt.json"))
    latest=ROOT/"data/rome_geometry/latest.json"
    bundle=ROOT/read_json(latest)["bundle"]
    if sha256_file(bundle/"manifest.json")!=feature_report["manifest_sha256"]:
        raise ValueError("Reviewed features changed.")
    members += [latest]+[p for p in bundle.rglob("*") if p.is_file()]
    files={}
    for path in sorted(set(members)):
        name=path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.is_symlink() or name.startswith("data/raw/") or ".env" in path.parts:
            raise ValueError("Unexpected source package member.")
        files[name]=sha256_file(path)
    with tarfile.open(package,"x:gz") as archive:
        for name in files: archive.add(ROOT/name,arcname=name,recursive=False)
    with tarfile.open(package,"r:gz") as archive:
        if len(archive.getmembers())!=len(files): raise ValueError("Source archive count differs.")
        for member in archive.getmembers():
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=files[member.name]:
                raise ValueError("Source archive readback differs.")
    receipt={"created_at_utc":utc_now(),"archive":package.relative_to(ROOT).as_posix(),"sha256":sha256_file(package),
        "size_bytes":package.stat().st_size,"files":files,"file_count":len(files),"model_code_sha256":declaration["model_code"]["sha256"],
        "feature_manifest_sha256":feature_report["manifest_sha256"],"scope":"Only reviewed new source, declarations, source receipts and the small feature extension; no original input Parquet, credentials or model changes."}
    write_json(receipt_path,receipt)
    remote_name=REMOTE+"/runs/rome-geometry-source-inputs.tar.gz"
    preflight=remote_python(f"""
import json,shutil,psutil
from pathlib import Path
root=Path({REMOTE!r}); destination=Path({remote_name!r})
if destination.exists(): raise RuntimeError('Preserve existing remote source archive')
active=[]
for p in psutil.process_iter(['pid','cmdline']):
 args=p.info['cmdline'] or []
 if any(x.startswith('tools/') and Path(x).name.startswith(('run_','refit_','batch_')) for x in args): active.append(p.info['pid'])
if active: raise RuntimeError('Another competition process is active')
free=shutil.disk_usage(root).free
if free < 850000000: raise RuntimeError('Insufficient declared disk margin')
print(json.dumps({{'free_disk_bytes':free,'active_project_training_pids':active}}))
""")
    subprocess.run(["scp","-q","-o","BatchMode=yes","-o","ConnectTimeout=15",str(package),"arcw0:"+remote_name],check=True)
    unpacked=remote_python(f"""
import hashlib,json,tarfile,shutil
from pathlib import Path
root=Path({REMOTE!r}).resolve(); path=Path({remote_name!r})
if hashlib.sha256(path.read_bytes()).hexdigest()!={receipt['sha256']!r}: raise RuntimeError('Remote package hash differs')
files=json.loads({json.dumps(files)!r}); added=[]; unchanged=[]
with tarfile.open(path,'r:gz') as archive:
 members=archive.getmembers()
 if len(members)!=len(files): raise RuntimeError('Archive member count differs')
 for member in members:
  target=(root/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(root): raise RuntimeError('Unexpected archive path')
  payload=archive.extractfile(member).read()
  if hashlib.sha256(payload).hexdigest()!=files[member.name]: raise RuntimeError('Archive payload hash differs')
  if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=files[member.name]: raise RuntimeError('Refuse to overwrite '+member.name)
 for member in members:
  target=root/member.name
  if target.exists(): unchanged.append(member.name); continue
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream: stream.write(archive.extractfile(member).read())
  added.append(member.name)
print(json.dumps({{'new_files':added,'existing_files_verified_unchanged':unchanged,'remote_archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'free_disk_bytes':shutil.disk_usage(root).free}}))
""")
    write_json(ROOT/"docs/rome_geometry/deployment.json",{"created_at_utc":utc_now(),"preflight":preflight,"unpacked":unpacked,
        "worker":"arcw0","package_sha256":receipt["sha256"],"audit_used":False})
    print(json.dumps({"deployed_files":len(unpacked["new_files"]),"existing_files_unchanged":len(unpacked["existing_files_verified_unchanged"]),
        "archive_bytes":receipt["size_bytes"],"remote_free_bytes":unpacked["free_disk_bytes"]}))


if __name__=="__main__": main()
