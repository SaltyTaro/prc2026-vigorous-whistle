"""Summarize saved Phase-5 error concentration; no training or label reads."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str):
    payload = (ROOT / relative).read_bytes()
    return json.loads(payload), {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    selection, selection_receipt = read("docs/phase5/development_results.json")
    standings, standings_receipt = read("docs/phase5/leaderboard_snapshot.json")
    assert selection["audit_used"] is False
    candidate = selection["candidates"][selection["selected_candidate"]]
    ours = standings["team_best"]["score"]
    third = standings["top_three"][2]["score"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [selection_receipt, standings_receipt],
        "selected_candidate": selection["selected_candidate"],
        "audit_used": False,
        "interpretation": (
            "Descriptive analysis of previously inspected complete development metrics. "
            "Slices overlap across grouping families and must not be added. "
            "Error concentration is not evidence that the errors can be predicted. "
            "No records are removed; no new model is trained or scored."
        ),
        "dated_official_gap": {
            "snapshot_at_utc": standings["retrieval_end_utc"],
            "our_score_seconds": ours,
            "third_place_seconds": third,
            "required_rmse_reduction_fraction": 1 - third / ours,
            "required_mse_reduction_fraction": 1 - (third / ours) ** 2,
            "not_a_final_cutoff": True,
        },
        "folds": {},
    }
    for fold, result in candidate["folds"].items():
        metrics = result["metrics"]
        overall = metrics["overall"]
        assert math.isclose(overall["rmse_seconds"] ** 2,
                            overall["sse"] / overall["rows"], rel_tol=1e-12)
        out = {"overall": overall, "evaluation_ids_sha256": metrics["evaluation_ids_sha256"]}
        for grouping in ["by_airport", "by_nm_missing", "by_target_band"]:
            groups = metrics[grouping]
            assert sum(s["rows"] for s in groups.values()) == overall["rows"]
            assert math.isclose(sum(s["sse"] for s in groups.values()),
                                overall["sse"], rel_tol=1e-12)
            out[grouping] = {
                name: {**value,
                       "row_fraction": value["rows"] / overall["rows"],
                       "sse_fraction": value["sse"] / overall["sse"]}
                for name, value in groups.items()
            }
        airports = [metrics["by_airport"][code] for code in ["LIRF", "LFPG"]]
        out["lirf_lfpg_combined"] = {
            "rows": sum(s["rows"] for s in airports),
            "row_fraction": sum(s["rows"] for s in airports) / overall["rows"],
            "sse_fraction": sum(s["sse"] for s in airports) / overall["sse"],
            "two_distinct_airport_maximum_errors_sse_fraction": sum(
                s["max_absolute_error_seconds"] ** 2 for s in airports
            ) / overall["sse"],
        }
        # Diagnostic upper bound, keeping every row in the RMSE denominator.
        # It assumes perfect predictions on the slice, not excluding that slice.
        share = out["by_target_band"]["over_3h"]["sse_fraction"]
        out["oracle_perfect_over_3h_only"] = {
            "all_rows_rmse_seconds": overall["rmse_seconds"] * math.sqrt(1 - share),
            "maximum_rmse_reduction_fraction": 1 - math.sqrt(1 - share),
            "achievable_forecast": False,
        }
        report["folds"][fold] = out
    path = ROOT / "docs/competition/history_20260914/error_headroom.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dated_official_gap": report["dated_official_gap"],
                      "folds": {k: {
                          "rows": v["overall"]["rows"],
                          "lirf_lfpg": v["lirf_lfpg_combined"],
                          "missing_nm": v["by_nm_missing"]["True"],
                          "over_3h": v["by_target_band"]["over_3h"],
                          "oracle_perfect_over_3h_only": v["oracle_perfect_over_3h_only"],
                      } for k, v in report["folds"].items()}}, indent=2))


if __name__ == "__main__":
    main()
