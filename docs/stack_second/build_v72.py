"""Build v72 from the verified stack_second_v1 dry run: bytes copied into runs/submissions, validated, receipt written.
Set USER_DECISION to the user's words before running."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402
from prc2026_submission.contract import validate_submission  # noqa: E402

DRY = ROOT / "runs/stack_second/rank/dry_v72/vigorous-whistle_v72.parquet"
OUT = ROOT / "runs/submissions/vigorous-whistle_v72.parquet"
SHA = "4280c399aa148847c4be4588dfc3d98d4adfd1a5e64981035b4e0fa3ee802f80"
USER_DECISION = "FILL IN the user's words before running"
if USER_DECISION.startswith("FILL IN"):
    raise ValueError("Build only at the user's word: fill in USER_DECISION.")
if OUT.exists() or sha256_file(DRY) != SHA:
    raise ValueError("Never overwrite; the dry run must be the verified file.")
OUT.write_bytes(DRY.read_bytes())
sub = validate_submission(OUT, ROOT / "data/raw/submitting.parquet")
if sub["sha256"] != SHA or not sub["exact_template_order"]:
    raise ValueError("Validation failed.")
sub.update({"version": 72, "provenance": {
    "audit_used": False, "official_scores_used": "none", "experiment_id": "stack_second_v1",
    "declaration": {"configs/stack_second_v1.json": sha256_file(ROOT / "configs/stack_second_v1.json")},
    "selected_by": "configs/stack_second_v1.json clauses 1-4 (development vs v71's values, transfer guard); lowest eq mean among passing; the user's word",
    "selected_candidate": "sp_hicov7", "base": "v71 (2dfb3b9f...)", "external_data": "none new",
    "user_decision": USER_DECISION, "dry_run": {"file": DRY.relative_to(ROOT).as_posix(), "sha256": SHA},
    "code": {p: sha256_file(ROOT / p) for p in ["tools/run_stack_second.py", "docs/stack_second/build_v72.py"]}}})
write_json(OUT.with_suffix(".receipt.json"), sub)
print(sub["sha256"], sub["rows"], sub["exact_template_order"])
