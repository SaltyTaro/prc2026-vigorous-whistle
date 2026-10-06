#!/usr/bin/env bash
# Tree own-points screen, local reading (configs/tree_own_screen_v1.json + configs/tree_own_screen_v1_amend1.json): control reproduction,
# tables, the unpinned zero-delta identity on v66's chain (must reproduce the v66 proxy), the pinned identity, then each declared candidate
# pinned through v66's development chain (model part -> chain -> score), one at a time; then docs/tree_own/screen_results.json.
# Run detached with a log; nothing is uploaded.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe -X utf8 -u"
[ -e runs/tree_own/tables.parquet ] || $PY tools/run_tree_own_chain.py --check --tables
[ -e runs/tree_own/chain_v66/identity/score.json ] || $PY tools/run_tree_own_chain.py --model-part identity --chain identity --score identity
[ -e runs/tree_own/chain_v66_pinned/identity/score.json ] || $PY tools/run_tree_own_chain.py --pinned --model-part identity --chain identity --score identity
for c in lirf_fam100 lirf_exact com_fam100 com_exact both_fam100; do
  [ -e runs/tree_own/chain_v66_pinned/$c/score.json ] || $PY tools/run_tree_own_chain.py --pinned --model-part $c --chain $c --score $c
done
PYTHONPATH=src $PY - <<'EOF'
import json
from pathlib import Path
from prc2026.artifacts import sha256_file, utc_now, write_json
root = Path(".")
P = root / "runs/tree_own/chain_v66_pinned"
out = {"created_at_utc": utc_now(), "declaration": "configs/tree_own_screen_v1.json", "declaration_sha256": sha256_file(root / "configs/tree_own_screen_v1.json"),
       "amendment": "configs/tree_own_screen_v1_amend1.json", "amendment_sha256": sha256_file(root / "configs/tree_own_screen_v1_amend1.json"),
       "control": "v66 development proxy", "check_passed": json.loads((root / "runs/tree_own/check.json").read_text())["passed"],
       "tables": json.loads((root / "runs/tree_own/tables.json").read_text()),
       "identity_unpinned_max_abs_vs_v66_s": json.loads((root / "runs/tree_own/chain_v66/identity/score.json").read_text())["max_abs_vs_v66_proxy_s"],
       "identity_pinned_max_abs_vs_v66_s": json.loads((P / "identity/score.json").read_text())["max_abs_vs_v66_proxy_s"], "candidates": {}}
for c in ["lirf_fam100", "lirf_exact", "com_fam100", "com_exact", "both_fam100"]:
    s = json.loads((P / c / "score.json").read_text())
    out["candidates"][c] = {"months": s["months"], "go_bar": s["go_bar"], "proxy": s["proxy"]}
out["go_candidates"] = [c for c, s in out["candidates"].items() if s["go_bar"]["GO"]]
out["best_by_equal_footing_mean"] = min(out["candidates"], key=lambda c: out["candidates"][c]["go_bar"]["equal_footing_mean_pct"])
Path("docs/tree_own").mkdir(parents=True, exist_ok=True)
write_json(root / "docs/tree_own/screen_results.json", out)
for c, s in out["candidates"].items():
    print(c, {m: round(v["pct"], 4) for m, v in s["months"].items()}, s["go_bar"])
print("GO:", out["go_candidates"])
EOF
