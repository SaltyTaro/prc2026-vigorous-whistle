"""Construct one versioned submission from a verified, audit-preserving refit."""

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026.artifacts import read_json, sha256_file, verify_receipts
from prc2026_submission.contract import export_submission


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit-run", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    run = root / args.refit_run
    metadata = read_json(run / "run.json")
    if metadata["stage"] != "refit" or metadata["audit_used"] or "2025-12" in metadata["refit_months"]:
        raise ValueError("Submission must come from the documented audit-preserving refit.")
    verify_receipts(run, metadata["artifacts"])
    template = root / "data/raw/submitting.parquet"
    manifest = read_json(root / "docs/competition/download_manifest.json")
    source = next(item for item in manifest["objects"] if item["key"] == "submitting.parquet")
    if sha256_file(template) != source["sha256"]:
        raise ValueError("Submission template differs from the audited input.")
    predictions = pd.read_parquet(run / "ranking_predictions.parquet")
    output = root / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    receipt = export_submission(predictions, template, output, {
        "refit_run_id": metadata["run_id"], "refit_run_sha256": sha256_file(run / "run.json"),
        "model_kind": metadata["model_kind"], "iterations": metadata["iterations"],
        "bundle_id": metadata["bundle_id"], "config_sha256": metadata["config_sha256"],
        "model_code_sha256": metadata["model_code"]["sha256"], "audit_used": False,
    })
    print(f"Built {output}: {receipt['rows']} rows; SHA256 {receipt['sha256']}")


if __name__ == "__main__":
    main()
