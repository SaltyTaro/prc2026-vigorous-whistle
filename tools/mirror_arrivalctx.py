"""Stream a hash-checked archive from arcw0, avoiding a second large VM copy."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,tarfile
root=Path('/home/azureuser/prc-flight-2026').resolve()
kind=__KIND__
bundle=__BUNDLE__
if kind=='features':
    base=root/'data/arrivalctx'/bundle
    manifest=json.loads((base/'manifest.json').read_text())
    assert manifest['bundle_id'].startswith(bundle) and not manifest['audit_used']
    files=[p for p in base.rglob('*') if p.is_file()]+[root/'data/arrivalctx/latest.json']
else:
    verification=json.loads((root/'docs/arrivalctx/verification.json').read_text())
    assert verification['passed'] and not verification['audit_used']
    completion=json.loads((root/'docs/arrivalctx/batch_completion.json').read_text())
    assert completion['declared_batch_complete']
    files=[p for base in [root/'runs/arrivalctx',root/'docs/arrivalctx'] for p in base.rglob('*') if p.is_file()]
receipt_path=root/'docs/arrivalctx'/(kind+'_archive_stream.json')
assert not receipt_path.exists()
files=sorted(set(files))
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
receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'kind':kind,'sha256':stream.digest.hexdigest(),'size_bytes':stream.size,'files':hashes,'file_count':len(files),'storage':'Streamed directly to the Windows workspace; original Azure directories are retained.','audit_used':False}
with receipt_path.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["features", "development"])
    parser.add_argument("--bundle", default="ad97a0d16cb9cec64782")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{20}", args.bundle):
        raise ValueError("Use the exact declared bundle prefix.")
    archive_path = ROOT/"runs"/f"arrivalctx-{args.kind}.tar.gz"
    partial = archive_path.with_name(archive_path.name+".incomplete")
    if archive_path.exists() or partial.exists():
        raise ValueError("Preserve the existing transfer and its receipt.")
    script = REMOTE.replace("__KIND__", repr(args.kind)).replace("__BUNDLE__", repr(args.bundle))
    with partial.open("xb") as output:
        process = subprocess.run(["ssh", "arcw0", "cd /home/azureuser/prc-flight-2026 && .venv/bin/python -"],
                                 input=script.encode("utf-8"), stdout=output, stderr=subprocess.PIPE, check=False)
    if process.returncode:
        raise RuntimeError("Archive transfer failed: "+process.stderr.decode("utf-8", errors="replace"))
    remote_receipt = f"docs/arrivalctx/{args.kind}_archive_stream.json"
    raw_receipt = subprocess.check_output(["ssh", "arcw0", "cat /home/azureuser/prc-flight-2026/"+remote_receipt])
    receipt = json.loads(raw_receipt)
    with partial.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != receipt["sha256"] or partial.stat().st_size != receipt["size_bytes"]:
        raise ValueError("Transferred bytes differ from the Azure stream digest.")
    if not partial.resolve().is_relative_to((ROOT/"runs").resolve()) or not archive_path.resolve().is_relative_to((ROOT/"runs").resolve()):
        raise ValueError("Archive path escaped the workspace.")
    partial.rename(archive_path)
    local_receipt = ROOT/remote_receipt
    if local_receipt.exists() and local_receipt.read_bytes() != raw_receipt:
        raise ValueError("Preserve differing local stream evidence.")
    if not local_receipt.exists():
        local_receipt.write_bytes(raw_receipt)
    snapshot = ROOT/"docs/arrivalctx/azure_snapshots"/digest[:16]
    restored = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(receipt["files"]) or len({m.name for m in members}) != len(members):
            raise ValueError("Unexpected or repeated archive members.")
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts or "\\" in member.name or member.name not in receipt["files"]:
                raise ValueError("Unsafe archive member.")
            if args.kind == "features":
                if member.name != "data/arrivalctx/latest.json" and not member.name.startswith(f"data/arrivalctx/{args.bundle}/"):
                    raise ValueError("Unexpected feature archive path.")
                base, relative = ROOT/"data/arrivalctx", member.name[len("data/arrivalctx/"):]
            elif member.name.startswith("runs/arrivalctx/"):
                base, relative = ROOT/"runs/azure_arcw0_arrivalctx", member.name[len("runs/arrivalctx/"):]
            elif member.name.startswith("docs/arrivalctx/"):
                base, relative = snapshot, member.name[len("docs/arrivalctx/"):]
            else:
                raise ValueError("Unexpected development archive path.")
            destination = base/Path(*PurePosixPath(relative).parts)
            if not destination.resolve().is_relative_to(base.resolve()):
                raise ValueError("Restore destination escaped its allowed directory.")
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != receipt["files"][member.name]:
                raise ValueError("A restored file differs from its producing-machine hash.")
            if destination.exists():
                with destination.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != receipt["files"][member.name]:
                        raise ValueError("Preserve differing local artifacts: "+str(destination))
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(payload)
            restored[member.name] = destination.relative_to(ROOT).as_posix()
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "archive": archive_path.relative_to(ROOT).as_posix(), "sha256": digest,
              "size_bytes": receipt["size_bytes"], "verified_files": len(restored), "destinations": restored}
    with (ROOT/"docs/arrivalctx"/f"{args.kind}_mirror.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ["passed", "archive", "sha256", "size_bytes", "verified_files"]}))


if __name__ == "__main__":
    main()
