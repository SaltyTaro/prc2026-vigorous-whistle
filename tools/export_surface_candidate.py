"""Prepare a versioned surface candidate after both final verifiers pass."""
from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, TARGET
from prc2026_nmclock.features import route_mask
from prc2026_submission.contract import FILENAME, filename_version, validate_submission
from prc2026_submission.remote import BUCKET, check_upload_window, list_team_objects
from prc2026_surface_refit.training import model_code_receipt
from prc_data import make_client
from summarize_nextlever_development import same_science


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=int)
    args = parser.parse_args()
    name = f"vigorous-whistle_v{args.version}.parquet"
    filename_version(name)
    output = ROOT / "runs/submissions" / name
    ready_path = ROOT / "docs/surface_refit" / f"submission_v{args.version}.json"
    if output.exists() or ready_path.exists():
        raise ValueError("Preserve every existing export and readiness receipt.")
    reports = [read_json(ROOT / "docs/surface_refit" / filename) for filename in ["verification.json", "local_verification.json"]]
    same_science(*reports)
    for report in reports:
        if (not report["passed"] or report["audit_used"] or report["complete_ranking_rows"] != 344841
                or report["fallback_rows_identical_to_v6"] != 5290
                or report["complete_native_reload_maximum_difference_seconds"] != 0.
                or report["independent_weight_maximum_difference_seconds"] != 0.
                or report["model_code_sha256"] != model_code_receipt(ROOT)["sha256"]):
            raise ValueError("Both complete final native verifications must pass without changes.")
    run = ROOT / "runs/azure_arcw0_surface_refit" / reports[0]["run_id"]
    if sha256_file(run / "run.json") != reports[0]["run_sha256"]:
        raise ValueError("The final verified refit changed.")
    metadata = read_json(run / "run.json")
    verify_receipts(run, metadata["artifacts"])
    client = make_client()
    objects = list_team_objects(client)
    local_versions = [filename_version(path.name) for path in (ROOT / "runs/submissions").glob("*.parquet") if FILENAME.fullmatch(path.name)]
    remote_versions = [filename_version(row["Key"]) for row in objects if FILENAME.fullmatch(row["Key"])]
    if args.version <= max([0, *local_versions, *remote_versions]):
        raise ValueError("Choose a fresh version above every local and live version.")
    inputs = read_json(ROOT / "docs/competition/download_manifest.json")["objects"]
    remote_inputs = {}
    for key in ["ranking.parquet", "submitting.parquet"]:
        source = next(row for row in inputs if row["key"] == key)
        current = client.head_object(Bucket="prc-2026-datasets", Key=key)
        etag = current["ETag"].strip('"')
        if etag != source["etag"] or current["ContentLength"] != source["size_bytes"]:
            raise ValueError("Official ranking or template inputs changed.")
        remote_inputs[key] = {"etag": etag, "size_bytes": current["ContentLength"], "local_sha256": source["sha256"]}
    before = {"checked_at_utc": utc_now(), "bucket": BUCKET, "input_objects": remote_inputs,
              "team_objects": [{"key": row["Key"], "size_bytes": row["Size"], "etag": row["ETag"].strip('"'), "last_modified_utc": row["LastModified"].isoformat()} for row in objects],
              "local_versions": sorted(local_versions), "remote_versions": sorted(remote_versions),
              "remote_write_performed": False}
    subprocess.run([sys.executable, "-B", "tools/build_submission.py", "--refit-run", run.relative_to(ROOT).as_posix(), "--version", str(args.version)], cwd=ROOT, check=True)
    validation = validate_submission(output, ROOT / "data/raw/submitting.parquet")
    export = read_json(output.with_suffix(".receipt.json"))
    frame = pd.read_parquet(output)
    features = pd.read_parquet(ROOT / "data/timeline/ce42240e5e43eee1713f/features/ranking.parquet", columns=[ID, "nmclock_available", "nmclock_proxy_s"])
    aligned = frame[[ID]].merge(features, on=ID, how="left", validate="one_to_one")
    missing = ~route_mask(aligned)
    previous = ROOT / "runs/submissions/vigorous-whistle_v6.parquet"
    if sha256_file(previous) != "b81c2030118a8b7693718a40ba4caf3e1e7d90103899d65f8c5a37a2cd5578e5":
        raise ValueError("The original v6 export changed.")
    v6 = frame[[ID]].merge(pd.read_parquet(previous), on=ID, how="left", validate="one_to_one")
    if int(missing.sum()) != 5290 or not np.array_equal(frame.loc[missing, TARGET], v6.loc[missing, TARGET]):
        raise ValueError("Export rounding must preserve all original missing-clock values.")
    try:
        limits = {"available": True, **check_upload_window(objects, name, output.stat().st_size)}
    except ValueError as exc:
        # A quota wait does not invalidate a locally verified immutable export.
        limits = {"available": False, "reason": str(exc)}
    report = {"created_at_utc": utc_now(), "status": "validated_export_ready", "filename": name,
              "sha256": validation["sha256"], "rows": validation["rows"], "size_bytes": validation["size_bytes"],
              "validation": validation, "export_receipt_sha256": sha256_file(output.with_suffix(".receipt.json")),
              "refit_run_id": run.name, "refit_run_sha256": sha256_file(run / "run.json"),
              "selection_sha256": metadata["selection_sha256"], "model_code_sha256": metadata["model_code"]["sha256"],
              "verification_sha256": {filename: sha256_file(ROOT / "docs/surface_refit" / filename) for filename in ["verification.json", "local_verification.json"]},
              "complete_native_reload_maximum_difference_seconds": 0., "independent_weight_maximum_difference_seconds": 0.,
              "rounded_missing_clock_values_identical_to_v6": 5290, "live_preflight": before,
              "upload_window_at_check": limits, "audit_used": False, "submitted": False,
              "official_result": None, "official_best_unchanged": {"version": 6, "rmse_seconds": 277.4508, "rows": 344841},
              "code_sha256": sha256_file(Path(__file__)), "export_rounding": export["rounding"]}
    write_json(ready_path, report)
    print({key: report[key] for key in ["status", "filename", "sha256", "rows", "rounded_missing_clock_values_identical_to_v6", "submitted"]})


if __name__ == "__main__":
    main()
