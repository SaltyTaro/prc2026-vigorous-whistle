"""Summarize v7's remaining error from frozen aggregate development metrics."""

from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json


def main():
    inputs = {
        name: ROOT / name for name in [
            "docs/surface_airports/development_results.json",
            "docs/surface_refit/submission_v7.json",
            "docs/surface_refit/submission_v7_official.json",
            "docs/surface_refit/leaderboard_snapshot.json",
            "docs/next_lever/development_summary.json",
            "docs/ladder/campaign_summary.json",
        ]
    }
    selection = read_json(inputs["docs/surface_airports/development_results.json"])
    ready = read_json(inputs["docs/surface_refit/submission_v7.json"])
    official = read_json(inputs["docs/surface_refit/submission_v7_official.json"])
    standings = read_json(inputs["docs/surface_refit/leaderboard_snapshot.json"])
    if sha256_file(inputs["docs/surface_airports/development_results.json"]) != ready["selection_sha256"]:
        raise ValueError("The frozen selection receipt changed.")
    if standings["team_best"] != official["official_result"]:
        raise ValueError("Official result and dated standings differ.")
    selected = selection["candidates"][selection["selected_candidate"]]
    score = official["official_rmse_seconds"]
    third = float(standings["top_three"][2]["score"])
    required_mse_fraction = 1 - (third / score) ** 2
    folds = {}
    for fold, entry in selected["folds"].items():
        metrics = entry["metrics"]
        overall = metrics["overall"]
        total = overall["sse"]
        if not math.isclose(overall["rmse_seconds"], math.sqrt(total / overall["rows"]), rel_tol=1e-12):
            raise ValueError("Saved RMSE and squared error disagree.")
        groups = {}
        for key in ["by_airport", "by_nm_missing", "by_target_band"]:
            if sum(row["rows"] for row in metrics[key].values()) != overall["rows"]:
                raise ValueError("Grouping does not cover the complete fold.")
            if not math.isclose(sum(row["sse"] for row in metrics[key].values()), total, rel_tol=1e-12):
                raise ValueError("Group squared errors do not sum to the complete fold.")
            groups[key] = {
                label: {**row, "row_fraction": row["rows"] / overall["rows"],
                        "sse_fraction": row["sse"] / total}
                for label, row in metrics[key].items()
            }
        airport_rows = [metrics["by_airport"][name] for name in ["LIRF", "LFPG"]]
        shares = {
            "missing_nm": groups["by_nm_missing"]["True"]["sse_fraction"],
            "nm_present": groups["by_nm_missing"]["False"]["sse_fraction"],
            "lirf_and_lfpg": sum(row["sse"] for row in airport_rows) / total,
            "target_0_to_1h": groups["by_target_band"]["0_to_1h"]["sse_fraction"],
            "target_over_3h": groups["by_target_band"]["over_3h"]["sse_fraction"],
        }
        scenarios = {}
        for name, share in shares.items():
            scenarios[name] = {
                "sse_fraction": share,
                "complete_rmse_reduction_if_slice_sse_falls_20pct": 1 - math.sqrt(1 - .2 * share),
                "complete_rmse_reduction_if_slice_sse_falls_50pct": 1 - math.sqrt(1 - .5 * share),
                "oracle_complete_rmse_reduction_if_slice_error_zero": 1 - math.sqrt(1 - share),
                "slice_sse_reduction_needed_for_dated_official_relative_gap": required_mse_fraction / share,
            }
        folds[fold] = {"overall": overall, "evaluation_ids_sha256": metrics["evaluation_ids_sha256"],
                       "groups": groups, "scenarios": scenarios,
                       "lirf_lfpg_row_fraction": sum(row["rows"] for row in airport_rows) / overall["rows"]}
    report = {
        "created_at_utc": utc_now(), "selected_candidate": selection["selected_candidate"],
        "official_rmse_seconds": score, "third_place_rmse_seconds": third,
        "leaderboard_snapshot_at_utc": standings["created_at_utc"],
        "required_official_rmse_reduction_fraction": 1 - third / score,
        "required_official_mse_reduction_fraction": required_mse_fraction,
        "ranking_missing_clock_rows": ready["rounded_missing_clock_values_identical_to_v6"],
        "ranking_rows": ready["rows"], "folds": folds,
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "code_sha256": sha256_file(Path(__file__)), "audit_used": False,
        "new_labels_read": False, "new_model_scores_computed": False,
        "interpretation": "Reused complete development metrics only. Overlapping slices are not additive. Slice improvements and oracle bounds are arithmetic scenarios, not achievable-gain forecasts. The hidden leaderboard error mix is unknown; the dated third-place score is not a final cutoff.",
    }
    output = ROOT / "docs/competition/gain_review_v7_20260915/error_headroom.json"
    if output.exists():
        raise ValueError("Preserve the original v7 gain review.")
    write_json(output, report)
    print(json.dumps({"required_mse_reduction_percent": 100 * required_mse_fraction,
                      "folds": {name: row["scenarios"] for name, row in folds.items()}}, indent=2))


if __name__ == "__main__":
    main()
