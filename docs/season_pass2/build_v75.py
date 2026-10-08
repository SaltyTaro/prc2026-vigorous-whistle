"""Build v75 from the verified season_pass_v2 dry run: bytes copied into runs/submissions, validated, receipt written.
Set USER_DECISION to the user's words before running."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402
from prc2026_submission.contract import validate_submission  # noqa: E402

DRY = ROOT / "runs/season_pass2/rank/dry_v75/vigorous-whistle_v75.parquet"
OUT = ROOT / "runs/submissions/vigorous-whistle_v75.parquet"
SHA = "a227c51ca90589bf6966f49ae94e2113394ee8f508bee0794d20871a800a6036"
USER_DECISION = "FILL IN: the user's words"
if USER_DECISION.startswith("FILL IN"):
    raise ValueError("Build only at the user's word: fill in USER_DECISION.")
if OUT.exists() or sha256_file(DRY) != SHA:
    raise ValueError("Never overwrite; the dry run must be the verified file.")
OUT.write_bytes(DRY.read_bytes())
sub = validate_submission(OUT, ROOT / "data/raw/submitting.parquet")
if sub["sha256"] != SHA or not sub["exact_template_order"]:
    raise ValueError("Validation failed.")
sub.update({"version": 75, "provenance": {
    "audit_used": False, "official_scores_used": "none", "experiment_id": "season_pass_v2",
    "declaration": {"configs/season_pass_v2.json": sha256_file(ROOT / "configs/season_pass_v2.json")},
    "selected_by": "configs/season_pass_v2.json clauses 1-3 (November / December vs v74's values, transfer guard); lowest November-December mean among passing; the user's word",
    "selected_candidate": "w25b_5", "base": "v74 (85e57d05...)", "external_data": "none new (adsb.lol features already in use)",
    "user_decision": USER_DECISION, "dry_run": {"file": DRY.relative_to(ROOT).as_posix(), "sha256": SHA},
    "code": {p: sha256_file(ROOT / p) for p in ["tools/run_season_pass2.py", "docs/season_pass2/build_v75.py"]}}})
write_json(OUT.with_suffix(".receipt.json"), sub)
print(sub["sha256"], sub["rows"], sub["exact_template_order"])
