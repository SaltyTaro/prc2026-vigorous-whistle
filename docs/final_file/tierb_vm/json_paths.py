"""Every repo file path named inside the given JSON files that exists locally and is missing / different on the VM."""
import json, os, re, sys
V, files = sys.argv[1], sys.argv[2:]
vm = {}
for line in open(f'{V}/vm_manifest.tsv', encoding='utf-8'):
    p, s = line.rstrip('\n').split('\t'); vm[p] = int(s)
out, tot = set(), 0
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items(): walk(k); walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
    elif isinstance(o, str) and re.match(r'^(runs|data|scratch)/', o):
        if os.path.isfile(o) and vm.get(o) != os.path.getsize(o): out.add(o)
        elif os.path.isdir(o):
            for d, _, fs in os.walk(o):
                for f in fs:
                    q = (d + "/" + f).replace(chr(92), "/")
                    if vm.get(q) != os.path.getsize(q): out.add(q)
for f in files:
    walk(json.load(open(f, encoding='utf-8')))
open(f'{V}/upj.txt', 'w', newline='\n').write(''.join(sorted(x + '\n' for x in out)))
print(len(out), 'files', round(sum(os.path.getsize(p) for p in out) / 1e9, 3), 'GB')
