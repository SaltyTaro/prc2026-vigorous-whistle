"""Build v76 from the verified key_memory_v1 dry run: bytes copied into runs/submissions, validated, receipt written.
Set USER_DECISION to the user's words before running."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402
from prc2026_submission.contract import validate_submission  # noqa: E402

DRY = ROOT / "runs/key_memory/rank/dry_v76/vigorous-whistle_v76.parquet"
OUT = ROOT / "runs/submissions/vigorous-whistle_v76.parquet"
SHA = "a2ff395db4fdf22be86e1001f5dd97c8f27adc7b836fd7b4902690a576f9725c"
USER_DECISION = "'upload v76 alone' (8 October 2026 UTC; km_9 on v75's dry run; v75 itself is never uploaded)"
if USER_DECISION.startswith("FILL IN"):
    raise ValueError("Build only at the user's word: fill in USER_DECISION.")
if OUT.exists() or sha256_file(DRY) != SHA:
    raise ValueError("Never overwrite; the dry run must be the verified file.")
OUT.write_bytes(DRY.read_bytes())
sub = validate_submission(OUT, ROOT / "data/raw/submitting.parquet")
if sub["sha256"] != SHA or not sub["exact_template_order"]:
    raise ValueError("Validation failed.")
sub.update({"version": 76, "provenance": {
    "audit_used": False, "official_scores_used": "none", "experiment_id": "key_memory_v1",
    "declaration": {"configs/key_memory_v1.json": sha256_file(ROOT / "configs/key_memory_v1.json")},
    "selected_by": "configs/key_memory_v1.json clauses 1-3 (November / December vs v75's values, transfer guard); lowest November-December mean among passing; the user's word",
    "selected_candidate": "km_9", "base": "v75 (a227c51c..., season_pass_v2 w25b_5)", "external_data": "none new (FLIGHT_mvt from the supplied data)",
    "user_decision": USER_DECISION, "dry_run": {"file": DRY.relative_to(ROOT).as_posix(), "sha256": SHA},
    "code": {p: sha256_file(ROOT / p) for p in ["tools/run_key_memory.py", "docs/key_memory/build_v76.py"]}}})
write_json(OUT.with_suffix(".receipt.json"), sub)
print(sub["sha256"], sub["rows"], sub["exact_template_order"])
