"""Markdown table of a Tier B readings file (docs/final_file/tierb_{model,chain}_results.json): change and paired-day interval per month."""
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
L = r["layers"]
def key(k):
    import re
    m = re.match(r"([A-Z])(\d+)([a-z]?)", k)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (k, 0, "")
print("| reading | layer | Feb 2025 change [95% interval] | rows moved | Jun 2025 change [95% interval] | rows moved | off? |")
print("|---|---|---|---:|---|---:|---|")
for k in sorted(L, key=key):
    v = L[k]; cells = []
    for m in ["2025-02", "2025-06"]:
        x = v["months"].get(m)
        if x is None: cells += ["not read", ""]; continue
        lo, hi = x["interval_pct"]
        cells += [f"{x['change_pct']:+.3f}% [{lo:+.3f}, {hi:+.3f}]", f"{x['rows_moved']:,}"]
    off = [m for m in ["2025-02", "2025-06"] if v["months"].get(m, {}).get("switch_off")]
    print(f"| {k} | {v.get('label', '')} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {', '.join(off) or 'no'} |")
print()
print("switched off:", r.get("switched_off"))
