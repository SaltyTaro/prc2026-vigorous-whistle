"""Build v71 as v70 was built: the verified dry-run bytes copied into runs/submissions, validated against the template, receipt written."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402
from prc2026_submission.contract import validate_submission  # noqa: E402

DRY = ROOT / "runs/nbr_state/rank/dry_v71/vigorous-whistle_v71.parquet"
OUT = ROOT / "runs/submissions/vigorous-whistle_v71.parquet"
SHA = "7a0b1431eef1146d7afd40c728bea7af6b90a411e257ca277564866e410b84ed"
USER_DECISION = "FILL IN the user's words before running"
if USER_DECISION.startswith("FILL IN"):
    raise ValueError("Build only at the user's word: fill in USER_DECISION.")
if OUT.exists() or sha256_file(DRY) != SHA:
    raise ValueError("Never overwrite; the dry run must be the verified file.")
OUT.write_bytes(DRY.read_bytes())
sub = validate_submission(OUT, ROOT / "data/raw/submitting.parquet")
if sub["sha256"] != SHA or not sub["exact_template_order"]:
    raise ValueError("Validation failed.")
sub.update({"version": 71, "provenance": {
    "audit_used": False, "official_scores_used": "none", "experiment_id": "nbr_state_v1",
    "declaration": {"configs/nbr_state_v1.json": sha256_file(ROOT / "configs/nbr_state_v1.json")},
    "selected_by": "configs/nbr_state_v1.json: clauses 1-5 (development, binding fresh August-October 2025 check, per-airport transfer guard: EHAM keeps v70); lowest eq mean among passing; the user's word",
    "selected_candidate": "w75", "base": "v70 (f1d17747...)",
    "external_data": "none new (adsb.lol globe_history matches already in the chain)",
    "user_decision": USER_DECISION,
    "dry_run": {"file": DRY.relative_to(ROOT).as_posix(), "sha256": SHA},
    "code": {p: sha256_file(ROOT / p) for p in ["tools/run_nbr_state.py",
                                                 "docs/nbr_state/build_v71.py"]}}})
write_json(OUT.with_suffix(".receipt.json"), sub)
print(sub["sha256"], sub["rows"], sub["exact_template_order"])
