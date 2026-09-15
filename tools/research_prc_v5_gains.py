"""Review v5 error concentration from saved metrics; no training or label reads."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/competition/gain_review_v5_20260914"


def read(relative: str, receipts: dict):
    payload = (ROOT / relative).read_bytes()
    receipts[relative] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def main() -> None:
    receipts = {}
    selection = read("docs/weather/development_results.json", receipts)
    standings = read("docs/weather/leaderboard_snapshot.json", receipts)
    field_review = read("docs/competition/field_use_review_20260914.json", receipts)
    earlier = read("docs/phase5r/development_results.json", receipts)
    assert not selection["audit_used"] and not earlier["audit_used"]
    assert selection["selected_candidate"] == "cat_weather_v4_100"
    assert earlier["selected_candidate"] is None
    candidate = selection["candidates"][selection["selected_candidate"]]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": receipts,
        "selected_candidate": selection["selected_candidate"],
        "audit_used": False,
        "new_labels_read": False,
        "new_model_trained_or_scored": False,
        "interpretation": (
            "Descriptive analysis of saved complete development metrics. "
            "Slices from different grouping families overlap and cannot be added. "
            "Error concentration does not establish predictability. Oracle values "
            "keep every row in the denominator and are not achievable forecasts."
        ),
        "dated_official_gap": {
            "snapshot_at_utc": standings["checked_at_utc"],
            "our_rmse_seconds": standings["team"]["score"],
            "third_place_rmse_seconds": standings["third_place"]["score"],
            "required_rmse_reduction_fraction": 1 - standings["third_place"]["score"] / standings["team"]["score"],
            "required_mse_reduction_fraction": 1 - (standings["third_place"]["score"] / standings["team"]["score"]) ** 2,
        },
        "field_use": {
            "finding": field_review["finding"],
            "prior_working_policy": field_review["working_policy"],
            "nm_offblock_ranking_present_rows": 344841 - 5290,
            "ranking_rows": 344841,
            "coverage_source": "docs/competition/DATA_SCHEMA.md",
            "clock_model_development_gain": "Not yet measured",
        },
        "folds": {},
    }
    schema = ROOT / "docs/competition/DATA_SCHEMA.md"
    receipts[str(schema.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(schema.read_bytes()).hexdigest()
    for fold, values in candidate["folds"].items():
        metrics = values["metrics"]
        overall = metrics["overall"]
        total = overall["sse"]
        assert math.isclose(overall["rmse_seconds"] ** 2, total / overall["rows"], rel_tol=1e-12)
        out = {"overall": overall, "evaluation_ids_sha256": metrics["evaluation_ids_sha256"]}
        for grouping in ("by_airport", "by_nm_missing", "by_target_band"):
            groups = metrics[grouping]
            assert sum(v["rows"] for v in groups.values()) == overall["rows"]
            assert math.isclose(sum(v["sse"] for v in groups.values()), total, rel_tol=1e-12)
            out[grouping] = {
                name: {**v, "row_fraction": v["rows"] / overall["rows"], "sse_fraction": v["sse"] / total}
                for name, v in groups.items()
            }
        airports = [metrics["by_airport"][a] for a in ("LIRF", "LFPG")]
        share = sum(v["sse"] for v in airports) / total
        out["lirf_lfpg_combined"] = {
            "rows": sum(v["rows"] for v in airports),
            "row_fraction": sum(v["rows"] for v in airports) / overall["rows"],
            "sse_fraction": share,
            "two_distinct_airport_maximum_errors_sse_fraction": sum(v["max_absolute_error_seconds"] ** 2 for v in airports) / total,
        }
        tail_share = out["by_target_band"]["over_3h"]["sse_fraction"]
        out["oracle_perfect_over_3h_only"] = {
            "all_rows_rmse_seconds": overall["rmse_seconds"] * math.sqrt(1 - tail_share),
            "maximum_rmse_reduction_fraction": 1 - math.sqrt(1 - tail_share),
        }
        out["hypothetical_halving_lirf_lfpg_squared_error"] = {
            "all_rows_rmse_seconds": overall["rmse_seconds"] * math.sqrt(1 - share / 2),
            "rmse_reduction_fraction": 1 - math.sqrt(1 - share / 2),
            "achievability_established": False,
        }
        report["folds"][fold] = out
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "error_headroom.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str((OUTPUT / "error_headroom.json").relative_to(ROOT)),
        "dated_official_gap": report["dated_official_gap"],
        "folds": {
            fold: {
                "lirf_lfpg_sse_fraction": v["lirf_lfpg_combined"]["sse_fraction"],
                "missing_nm_sse_fraction": v["by_nm_missing"]["True"]["sse_fraction"],
                "over_three_hour_sse_fraction": v["by_target_band"]["over_3h"]["sse_fraction"],
                "oracle_over_three_hour_rmse_reduction": v["oracle_perfect_over_3h_only"]["maximum_rmse_reduction_fraction"],
            } for fold, v in report["folds"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
