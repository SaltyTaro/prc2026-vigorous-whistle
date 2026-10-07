"""Build v68 as v66 was built: the verified dry-run bytes copied into runs/submissions, validated against the template, receipt written."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402
from prc2026_submission.contract import validate_submission  # noqa: E402

DRY = ROOT / "runs/opdi_stack_lszh/rank/dry_v68/vigorous-whistle_v68.parquet"
OUT = ROOT / "runs/submissions/vigorous-whistle_v68.parquet"
SHA = "d41d8705e75a71e865fe21a8d6ad2f62044a5ed07f125fd208067914b2b10eab"
if OUT.exists() or sha256_file(DRY) != SHA:
    raise ValueError("Never overwrite; the dry run must be the verified file.")
OUT.write_bytes(DRY.read_bytes())
sub = validate_submission(OUT, ROOT / "data/raw/submitting.parquet")
if sub["sha256"] != SHA or not sub["exact_template_order"]:
    raise ValueError("Validation failed.")
sub.update({"version": 68, "provenance": {
    "audit_used": False, "official_scores_used": "none", "experiment_id": "opdi_stack_lszh_v1",
    "declaration": {"configs/opdi_stack_lszh_v1.json": sha256_file(ROOT / "configs/opdi_stack_lszh_v1.json"),
                    "configs/opdi_stack_v1.json": sha256_file(ROOT / "configs/opdi_stack_v1.json")},
    "selected_by": "configs/opdi_stack_lszh_v1.json clauses 1-3 and the LSZH transfer guard (post-hoc LSZH restriction, disclosed); the user's word",
    "selected_candidate": "opdi_lszh", "base": "v66 (b4ea727e...)",
    "user_decision": "'run it here itself' then 'build and upload it' (6-7 October 2026)",
    "dry_run": {"file": DRY.relative_to(ROOT).as_posix(), "sha256": SHA},
    "code": {p: sha256_file(ROOT / p) for p in ["tools/run_opdi_stack_lszh.py", "tools/run_opdi_stack_decl.py", "tools/run_opdi_stack.py",
                                                 "tools/run_stack_resid_v3.py", "tools/run_own_motion_v2.py", "docs/opdi_stack_lszh/build_v68.py"]}}})
write_json(OUT.with_suffix(".receipt.json"), sub)
print(sub["sha256"], sub["rows"], sub["exact_template_order"])
