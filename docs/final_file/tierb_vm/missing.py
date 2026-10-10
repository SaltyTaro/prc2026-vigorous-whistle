"""Workstation side: from the VM's .enoent / .dirs files of a step and a fresh VM manifest, list the files to upload next.
usage: missing.py OUT_LIST VM_MANIFEST ENOENT [DIRS ...]  (DIRS files: directories the step listed; their local direct children missing on the VM are added)"""
import os, sys
out, vm_manifest, enoent, dirs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
vm = {}
for line in open(vm_manifest, encoding="utf-8"):
    p, s = line.rstrip("\n").split("\t"); vm[p] = int(s)
want, why = {}, {}
def add_file(p, reason):
    if os.path.isfile(p) and vm.get(p) != os.path.getsize(p):
        want[p] = os.path.getsize(p); why.setdefault(reason, 0); why[reason] += 1
for line in open(enoent, encoding="utf-8"):
    p = line.strip()
    if not p: continue
    if os.path.isfile(p): add_file(p, "enoent-file")
    elif os.path.isdir(p):
        for d, _, fs in os.walk(p, followlinks=True):
            for f in fs: add_file(os.path.join(d, f).replace("\\", "/"), "enoent-dir")
for df in dirs:
    for line in open(df, encoding="utf-8"):
        d = line.strip().rstrip("/")
        if d and os.path.isdir(d):
            for f in os.listdir(d):
                q = f"{d}/{f}"
                if os.path.isfile(q): add_file(q, "listed-dir")
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    for p in sorted(want): fh.write(p + "\n")
print(len(want), "files", round(sum(want.values()) / 1e9, 3), "GB", why)
