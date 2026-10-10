"""Markdown table of docs/final_file/tierb_chain_results.json on the governing rows (lane M's rows read)."""
import json, re, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
def key(k):
    m = re.match(r"C(\d+)", k); return int(m.group(1)) if m else 99
print("| reading | Feb 2025 change [95% interval, s] | rows moved | Jun 2025 change [95% interval, s] | rows moved | off? |")
print("|---|---|---:|---|---:|---|")
for k in sorted(r["layers"], key=key):
    v = r["layers"][k]; cells = []
    for m in ["2025-02", "2025-06"]:
        g = (v["months"].get(m) or {}).get("governing_lane_m_rows")
        if not g: cells += ["not read", ""]; continue
        cells += [f"{g['change_pct']:+.3f}% [{g['lo_s']:+.3f}, {g['hi_s']:+.3f}]", f"{g['rows_moved']:,}"]
    off = [m for m in r.get("switched_off", {}) if k in r["switched_off"][m]]
    print(f"| {k} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {', '.join(off) or 'no'} |")
print()
print("switched off:", r.get("switched_off"), "; governing rows:", {m: x for m, x in r.get("rows", {}).items()} if isinstance(r.get("rows"), dict) else r.get("rows"))
