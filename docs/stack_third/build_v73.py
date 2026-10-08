"""Build v73 from the verified stack_third_v1 dry run: bytes copied into runs/submissions, validated, receipt written.
Set USER_DECISION to the user's words before running."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402
from prc2026_submission.contract import validate_submission  # noqa: E402

DRY = ROOT / "runs/stack_third/rank/dry_v73/vigorous-whistle_v73.parquet"
OUT = ROOT / "runs/submissions/vigorous-whistle_v73.parquet"
SHA = "649e99725ee3f700041b07a231dcd2fa656fdbeef3aa8db5b5b9c9fc341af3ba"
USER_DECISION = "FILL IN the user's words before running"
if USER_DECISION.startswith("FILL IN"):
    raise ValueError("Build only at the user's word: fill in USER_DECISION.")
if OUT.exists() or sha256_file(DRY) != SHA:
    raise ValueError("Never overwrite; the dry run must be the verified file.")
OUT.write_bytes(DRY.read_bytes())
sub = validate_submission(OUT, ROOT / "data/raw/submitting.parquet")
if sub["sha256"] != SHA or not sub["exact_template_order"]:
    raise ValueError("Validation failed.")
sub.update({"version": 73, "provenance": {
    "audit_used": False, "official_scores_used": "none", "experiment_id": "stack_third_v1",
    "declaration": {"configs/stack_third_v1.json": sha256_file(ROOT / "configs/stack_third_v1.json")},
    "selected_by": "configs/stack_third_v1.json clauses 1-4 (development vs v72's values, transfer guard); lowest eq mean among passing; the user's word",
    "selected_candidate": "t3_ro_3", "base": "v72 (4280c399...)", "external_data": "none new (adsb.lol features already in use)",
    "user_decision": USER_DECISION, "dry_run": {"file": DRY.relative_to(ROOT).as_posix(), "sha256": SHA},
    "code": {p: sha256_file(ROOT / p) for p in ["tools/run_stack_third.py", "docs/stack_third/build_v73.py"]}}})
write_json(OUT.with_suffix(".receipt.json"), sub)
print(sub["sha256"], sub["rows"], sub["exact_template_order"])
