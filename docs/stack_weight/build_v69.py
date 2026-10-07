"""Build v69 as v68 was built: the verified dry-run bytes copied into runs/submissions, validated against the template, receipt written."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402
from prc2026_submission.contract import validate_submission  # noqa: E402

DRY = ROOT / "runs/stack_weight/rank/dry_v69/vigorous-whistle_v69.parquet"
OUT = ROOT / "runs/submissions/vigorous-whistle_v69.parquet"
SHA = "192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665"
if OUT.exists() or sha256_file(DRY) != SHA:
    raise ValueError("Never overwrite; the dry run must be the verified file.")
OUT.write_bytes(DRY.read_bytes())
sub = validate_submission(OUT, ROOT / "data/raw/submitting.parquet")
if sub["sha256"] != SHA or not sub["exact_template_order"]:
    raise ValueError("Validation failed.")
sub.update({"version": 69, "provenance": {
    "audit_used": False, "official_scores_used": "none", "experiment_id": "stack_weight_v1",
    "declaration": {"configs/stack_weight_v1.json": sha256_file(ROOT / "configs/stack_weight_v1.json")},
    "selected_by": "configs/stack_weight_v1.json: v57's clauses 1-3 and the transfer guard; the only accepted candidate; the user's word",
    "selected_candidate": "w65_bag10", "base": "v68 (d41d8705...)",
    "user_decision": "'build and upload v69' (7 October 2026)",
    "dry_run": {"file": DRY.relative_to(ROOT).as_posix(), "sha256": SHA},
    "code": {p: sha256_file(ROOT / p) for p in ["tools/run_stack_weight.py", "tools/run_opdi_stack_decl.py", "tools/run_opdi_stack.py",
                                                 "tools/run_stack_resid_v3.py", "tools/run_stack_resid.py", "tools/run_own_motion_v2.py",
                                                 "docs/stack_weight/build_v69.py"]}}})
write_json(OUT.with_suffix(".receipt.json"), sub)
print(sub["sha256"], sub["rows"], sub["exact_template_order"])
