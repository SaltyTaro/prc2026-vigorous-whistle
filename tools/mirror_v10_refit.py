"""Stream the hash-checked v10 refit run from arcw0 into runs/azure_arcw0_v10/."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,tarfile
root=Path('/home/azureuser/prc-flight-2026').resolve()
runs=[p.parent for p in (root/'runs/v10').glob('*_refit_*/run.json') if json.loads(p.read_text())['stage']=='refit']
assert len(runs)==1, runs
run=runs[0]
report=json.loads((run/'run.json').read_text())
assert not report['audit_used'] and report['max_model_reload_difference_seconds']<=1e-8
files=sorted([p for p in run.rglob('*') if p.is_file()]+[root/'docs/v10/refit_launch.json',root/'docs/v10/tests_azure.xml',root/'runs/v10/refit.log'])
receipt_path=root/'docs/v10/refit_archive_stream.json'
assert not receipt_path.exists()
def file_hash(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
hashes={p.relative_to(root).as_posix():file_hash(p) for p in files}
for p in files:assert not p.is_symlink() and p.resolve().is_relative_to(root)
class HashedOutput:
    def __init__(self):self.digest=hashlib.sha256();self.size=0
    def write(self,data):
        sys.stdout.buffer.write(data);self.digest.update(data);self.size+=len(data);return len(data)
    def flush(self):sys.stdout.buffer.flush()
stream=HashedOutput()
with tarfile.open(fileobj=stream,mode='w|gz',compresslevel=3) as archive:
    for p in files:archive.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
stream.flush()
for name,digest in hashes.items():assert file_hash(root/name)==digest,('File changed during transfer',name)
receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'kind':'refit','run_id':run.name,'sha256':stream.digest.hexdigest(),'size_bytes':stream.size,'files':hashes,'file_count':len(files),'storage':'Streamed directly to the Windows workspace; original Azure directories are retained.','audit_used':False}
with receipt_path.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
'''


def main():
    archive_path = ROOT / "runs/v10-refit.tar.gz"
    partial = archive_path.with_name(archive_path.name + ".incomplete")
    if archive_path.exists() or partial.exists():
        raise ValueError("Preserve the existing transfer and its receipt.")
    with partial.open("xb") as output:
        process = subprocess.run(["ssh", "arcw0", "cd /home/azureuser/prc-flight-2026 && .venv/bin/python -"],
                                 input=REMOTE.encode("utf-8"), stdout=output, stderr=subprocess.PIPE, check=False)
    if process.returncode:
        raise RuntimeError("Archive transfer failed: " + process.stderr.decode("utf-8", errors="replace"))
    remote_receipt = "docs/v10/refit_archive_stream.json"
    raw_receipt = subprocess.check_output(["ssh", "arcw0", "cat /home/azureuser/prc-flight-2026/" + remote_receipt])
    receipt = json.loads(raw_receipt)
    with partial.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != receipt["sha256"] or partial.stat().st_size != receipt["size_bytes"]:
        raise ValueError("Transferred bytes differ from the Azure stream digest.")
    partial.rename(archive_path)
    local_receipt = ROOT / remote_receipt
    if local_receipt.exists() and local_receipt.read_bytes() != raw_receipt:
        raise ValueError("Preserve differing local stream evidence.")
    if not local_receipt.exists():
        local_receipt.write_bytes(raw_receipt)
    snapshot = ROOT / "docs/v10/azure_snapshots" / digest[:16]
    restored = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(receipt["files"]) or len({m.name for m in members}) != len(members):
            raise ValueError("Unexpected or repeated archive members.")
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts or "\\" in member.name or member.name not in receipt["files"]:
                raise ValueError("Unsafe archive member.")
            if member.name.startswith("runs/v10/"):
                base, relative = ROOT / "runs/azure_arcw0_v10", member.name[len("runs/v10/"):]
            elif member.name.startswith("docs/v10/"):
                base, relative = snapshot, member.name[len("docs/v10/"):]
            else:
                raise ValueError("Unexpected archive path.")
            destination = base / Path(*PurePosixPath(relative).parts)
            if not destination.resolve().is_relative_to(base.resolve()):
                raise ValueError("Restore destination escaped its allowed directory.")
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != receipt["files"][member.name]:
                raise ValueError("A restored file differs from its producing-machine hash.")
            if destination.exists():
                with destination.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != receipt["files"][member.name]:
                        raise ValueError("Preserve differing local artifacts: " + str(destination))
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(payload)
            restored[member.name] = destination.relative_to(ROOT).as_posix()
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "run_id": receipt["run_id"],
              "archive": archive_path.relative_to(ROOT).as_posix(), "sha256": digest,
              "size_bytes": receipt["size_bytes"], "verified_files": len(restored), "destinations": restored}
    with (ROOT / "docs/v10/refit_mirror.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ["passed", "run_id", "archive", "sha256", "size_bytes", "verified_files"]}))


if __name__ == "__main__":
    main()
