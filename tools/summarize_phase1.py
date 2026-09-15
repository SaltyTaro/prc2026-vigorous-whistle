"""Verify collected pilot artifacts and save a portable phase-1 evidence report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026.artifacts import read_json, utc_now, verify_receipts, write_json


def select_runs(path: Path, bundle_id: str) -> dict:
    selected = {}
    for filename in sorted(path.glob("*/run.json")):
        run = read_json(filename)
        if run["bundle_id"] != bundle_id:
            continue
        if run["fold"] not in {"seasonal", "forward"}:
            raise ValueError("Unexpected audit score in phase-1 evidence.")
        verify_receipts(filename.parent, run["artifacts"])
        selected[run["fold"]] = (filename.parent, run)
    if set(selected) != {"seasonal", "forward"}:
        raise ValueError("Both development evaluations are required.")
    return selected


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-bundle", type=Path, required=True)
    parser.add_argument("--azure-manifest", type=Path,
                        default=Path("infra/azure/.local/arcw0_feature_manifest.json"))
    parser.add_argument("--local-runs", type=Path, default=Path("runs/phase1"))
    parser.add_argument("--azure-runs", type=Path, default=Path("runs/azure_arcw0_phase1"))
    parser.add_argument("--output", type=Path, default=Path("docs/phase1"))
    args = parser.parse_args()
    local_base = root / args.local_bundle
    local = read_json(local_base / "manifest.json")
    azure = read_json(root / args.azure_manifest)
    verify_receipts(local_base, local["artifacts"])
    if local["bundle_id"] != azure["bundle_id"] or local["folds"] != azure["folds"]:
        raise ValueError("Platform bundles or frozen folds differ.")
    a = {item["path"]: item["sha256"] for item in local["artifacts"]}
    b = {item["path"]: item["sha256"] for item in azure["artifacts"]}
    if a != b:
        raise ValueError("Feature/split artifact hashes differ across platforms.")
    local_runs = select_runs(root / args.local_runs, local["bundle_id"])
    azure_runs = select_runs(root / args.azure_runs, azure["bundle_id"])
    comparisons, benchmarks = {}, {}
    for fold in ["seasonal", "forward"]:
        lp, lr = local_runs[fold]
        ap, ar = azure_runs[fold]
        left = pd.read_parquet(lp / "predictions.parquet").sort_values("MVT_ID_mvt")
        right = pd.read_parquet(ap / "predictions.parquet").sort_values("MVT_ID_mvt")
        if not np.array_equal(left["MVT_ID_mvt"], right["MVT_ID_mvt"]):
            raise ValueError("Cross-platform prediction IDs differ.")
        delta = float(np.max(np.abs(left["prediction"].to_numpy() - right["prediction"].to_numpy())))
        if delta > 1e-9:
            raise ValueError("Cross-platform predictions differ by more than 1e-9 seconds.")
        lm, am = read_json(lp / "metrics.json"), read_json(ap / "metrics.json")
        if not np.isclose(lm["overall"]["rmse_seconds"], am["overall"]["rmse_seconds"], rtol=1e-12):
            raise ValueError("Cross-platform RMSE differs.")
        comparisons[fold] = {"prediction_rows": len(left), "max_prediction_difference_seconds": delta,
                             "rmse_matches_within_relative_tolerance": 1e-12}
        tail = am["by_target_band"].get("over_3h", {"rows": 0, "sse": 0})
        benchmarks[fold] = {
            "model": "fit-partition airport arithmetic means", "official_submission": False,
            "local_run": lr, "azure_run": ar, "metrics": am,
            "labels_over_3h": {"rows": tail["rows"],
                              "share_of_total_squared_error": tail["sse"] / am["overall"]["sse"]},
        }
    output = root / args.output
    validation = {key: local[key] for key in [
        "bundle_id", "descriptor", "code", "folds", "feature_contract",
        "training_departures", "ranking_departures", "training_ids_sha256",
        "ranking_ids_sha256", "ranking_template_order_matches", "ranking_month_counts",
        "ranking_month_weights",
    ]}
    validation["filename_month_spills"] = [
        {"source": source["source"], "counts": source["rows_outside_filename_month_by_phase"]}
        for source in local["sources"] if source["rows_outside_filename_month_by_phase"]
    ]
    validation["artifacts"] = local["artifacts"]
    write_json(output / "validation_report.json", validation)
    summary = {
        "created_at_utc": utc_now(), "bundle_id": local["bundle_id"],
        "matching_feature_and_split_hashes": len(a),
        "materialized_artifact_bytes": sum(item["size_bytes"] for item in local["artifacts"]),
        "local_preparation": local["preparation"], "azure_preparation": azure["preparation"],
        "cross_platform_comparison": comparisons, "benchmarks": benchmarks,
        "audit_status": "December reserved; no December model evaluation in phase 1",
    }
    write_json(output / "benchmark_summary.json", summary)
    print(f"Verified {len(a)} matching artifacts and both development prediction sets.")


if __name__ == "__main__":
    main()
