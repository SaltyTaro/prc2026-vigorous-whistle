#!/usr/bin/env bash
# Tree own-points screen, local reading (configs/tree_own_screen_v1.json): control reproduction, tables, then each declared candidate through
# v65's development chain (model part -> chain -> score), one at a time; then the collected table docs/tree_own/screen_results.json.
# Run detached with a log; nothing is uploaded.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe -X utf8 -u"
$PY tools/run_tree_own_chain.py --check --tables
for c in lirf_fam100 lirf_exact com_fam100 com_exact both_fam100; do
  if [ ! -e runs/tree_own/chain/$c/score.json ]; then
    $PY tools/run_tree_own_chain.py --model-part $c --chain $c --score $c
  fi
done
PYTHONPATH=src $PY - <<'EOF'
import json
from pathlib import Path
from prc2026.artifacts import sha256_file, utc_now, write_json
root = Path(".")
out = {"created_at_utc": utc_now(), "declaration": "configs/tree_own_screen_v1.json", "declaration_sha256": sha256_file(root / "configs/tree_own_screen_v1.json"),
       "check": json.loads((root / "runs/tree_own/check.json").read_text())["passed"], "tables": json.loads((root / "runs/tree_own/tables.json").read_text()),
       "identity_max_abs_vs_v65_proxy_s": json.loads((root / "runs/tree_own/chain/identity/score.json").read_text())["max_abs_vs_v65_proxy_s"], "candidates": {}}
for c in ["lirf_fam100", "lirf_exact", "com_fam100", "com_exact", "both_fam100"]:
    s = json.loads((root / f"runs/tree_own/chain/{c}/score.json").read_text())
    out["candidates"][c] = {"months": s["months"], "go_bar": s["go_bar"], "proxy": s["proxy"]}
go = [c for c, s in out["candidates"].items() if s["go_bar"]["GO"]]
out["go_candidates"] = go
out["best_by_equal_footing_mean"] = min(out["candidates"], key=lambda c: out["candidates"][c]["go_bar"]["equal_footing_mean_pct"])
Path("docs/tree_own").mkdir(parents=True, exist_ok=True)
write_json(root / "docs/tree_own/screen_results.json", out)
for c, s in out["candidates"].items():
    print(c, {m: round(v["pct"], 4) for m, v in s["months"].items()}, s["go_bar"])
print("GO:", go)
EOF
