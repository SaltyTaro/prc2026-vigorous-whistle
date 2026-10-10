import json, os, sys
V = sys.argv[1]
plan = json.load(open('runs/febjun_lomo/plan.json'))
leaves = sorted({j['leaf'] for h in plan['chains'].values() for j in h})
vm = {}
for line in open(f'{V}/vm_manifest.tsv', encoding='utf-8'):
    p, s = line.rstrip('\n').split('\t'); vm[p] = int(s)
out = set(l.strip() for l in open(f'{V}/up2.txt') if '_standin/' not in l)
for leaf in leaves:
    for d in [leaf, leaf + '/refit']:
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            p = d + '/' + f
            if os.path.isfile(p) and p.endswith('.json') and vm.get(p) != os.path.getsize(p): out.add(p)
open(f'{V}/up2b.txt', 'w', newline='\n').write(''.join(sorted(x + '\n' for x in out)))
print(len(leaves), 'leaves;', len(out), 'files to send')
