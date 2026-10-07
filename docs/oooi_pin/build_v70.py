"""Build v70 as v69 was built: the verified dry-run bytes copied into runs/submissions, validated against the template, receipt written."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402
from prc2026_submission.contract import validate_submission  # noqa: E402

DRY = ROOT / "runs/oooi_pin/rank/dry_v70/vigorous-whistle_v70.parquet"
OUT = ROOT / "runs/submissions/vigorous-whistle_v70.parquet"
SHA = "f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f"
if OUT.exists() or sha256_file(DRY) != SHA:
    raise ValueError("Never overwrite; the dry run must be the verified file.")
OUT.write_bytes(DRY.read_bytes())
sub = validate_submission(OUT, ROOT / "data/raw/submitting.parquet")
if sub["sha256"] != SHA or not sub["exact_template_order"]:
    raise ValueError("Validation failed.")
sub.update({"version": 70, "provenance": {
    "audit_used": False, "official_scores_used": "none", "experiment_id": "oooi_pin_v1",
    "declaration": {"configs/oooi_pin_v1.json": sha256_file(ROOT / "configs/oooi_pin_v1.json")},
    "selected_by": "configs/oooi_pin_v1.json: clauses 1-5 (development, binding fresh August-October 2025 check, transfer guard); the only accepted candidate; the user's word",
    "selected_candidate": "w50", "base": "v69 (192c4a30...)",
    "external_data": "adsb.lol Aircraft Data Links archive (CC0 1.0), github.com/adsblol/aircraft-data-links-2025 and -2026; receipts docs/oooi_pin/receipts/",
    "user_decision": "'yes build and upload now' (7 October 2026)",
    "dry_run": {"file": DRY.relative_to(ROOT).as_posix(), "sha256": SHA},
    "code": {p: sha256_file(ROOT / p) for p in ["tools/run_oooi_pin.py", "tools/oooi_pin_join.py", "tools/oooi_pin_extract.py",
                                                 "tools/oooi_pin_pack.py", "docs/oooi_pin/build_v70.py"]}}})
write_json(OUT.with_suffix(".receipt.json"), sub)
print(sub["sha256"], sub["rows"], sub["exact_template_order"])
