"""Build the next version (v71) from a verified nbr_state dry run: bytes copied into runs/submissions, validated, receipt written.

CHOICE "state"      = v70 + nbr_state_v1 w75 (unmatched rows)                  -> dry run 7a0b1431...
CHOICE "state+dark" = v70 + nbr_state_v1 w75 + nbr_dark_v1 (dark rows as well) -> dry run 2dfb3b9f...
Both are accepted by their declared rules; set CHOICE and USER_DECISION (the user's words) before running.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402
from prc2026_submission.contract import validate_submission  # noqa: E402

DRIES = {
    "state": ("runs/nbr_state/rank/dry_v71/vigorous-whistle_v71.parquet",
              "7a0b1431eef1146d7afd40c728bea7af6b90a411e257ca277564866e410b84ed", ["nbr_state_v1"], ["w75"]),
    "state+dark": ("runs/nbr_state/rank/dry_v72/vigorous-whistle_v72.parquet",
                   "2dfb3b9f53284c7ae3a02882d19cfd61b19116b1b00504176551bcae8b82e6ba", ["nbr_state_v1", "nbr_dark_v1"], ["w75", "dark_pool_w75"]),
}
CHOICE = "state+dark"
USER_DECISION = "'upload it' (8 October 2026 UTC), after the combined file (nbr_state + nbr_dark) was offered as v71"
VERSION = 71
if CHOICE not in DRIES or USER_DECISION.startswith("FILL IN"):
    raise ValueError("Build only at the user's word: set CHOICE and USER_DECISION.")
dry, SHA, experiments, selected = DRIES[CHOICE]
DRY = ROOT / dry
OUT = ROOT / f"runs/submissions/vigorous-whistle_v{VERSION}.parquet"
if OUT.exists() or sha256_file(DRY) != SHA:
    raise ValueError("Never overwrite; the dry run must be the verified file.")
OUT.write_bytes(DRY.read_bytes())
sub = validate_submission(OUT, ROOT / "data/raw/submitting.parquet")
if sub["sha256"] != SHA or not sub["exact_template_order"]:
    raise ValueError("Validation failed.")
sub.update({"version": VERSION, "provenance": {
    "audit_used": False, "official_scores_used": "none", "experiment_id": experiments,
    "declaration": {f"configs/{e}.json": sha256_file(ROOT / f"configs/{e}.json") for e in experiments},
    "selected_by": "each declaration's clauses (development, binding fresh August-October 2025 check, per-airport transfer guard); the user's word",
    "selected_candidate": selected, "base": "v70 (f1d17747...)",
    "external_data": "none new (adsb.lol globe_history matches already in the chain)",
    "user_decision": USER_DECISION,
    "dry_run": {"file": dry, "sha256": SHA},
    "code": {p: sha256_file(ROOT / p) for p in ["tools/run_nbr_state.py", "tools/run_nbr_dark.py", "docs/nbr_state/build_v71.py"]}}})
write_json(OUT.with_suffix(".receipt.json"), sub)
print(sub["sha256"], sub["rows"], sub["exact_template_order"])
