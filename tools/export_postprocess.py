"""Export the selected post-processing candidate as the next versioned submission.

Applies the selected rules to the frozen v7 ranking predictions, with the frozen
no-clock (v5) ranking predictions as the routing fallback, using parameters fitted
on the refit months only. Every untouched row must reproduce the uploaded v7
integer exactly. Nothing is uploaded.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET
from prc2026.validation import identity_hash
from prc2026_postprocess.rules import POLICY, RAW_COLUMNS, apply_rules, fit_parameters, row_context
from prc2026_submission.contract import export_submission
from run_postprocess import check_parents, departures, model_code_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/postprocess_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/postprocess/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/postprocess/export.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    selection = read_json(ROOT / args.selection)
    if selection["config_sha256"] != object_hash(config) or selection["audit_used"] or selection["native_fit_count"] != 0:
        raise ValueError("Selection does not belong to this declaration.")
    if selection["model_code"]["sha256"] != model_code_receipt(ROOT)["sha256"]:
        raise ValueError("Scientific code changed after selection.")
    check_parents(config)
    rules = selection["selected_rules"]
    if rules != config["candidates"][selection["selected_candidate"]]:
        raise ValueError("Selected rules differ from the declared candidate.")
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=RAW_COLUMNS + [PHASE])
    ranking = ranking.loc[ranking[PHASE].eq("DEP")].drop(columns=[PHASE]).reset_index(drop=True)
    incumbent = pd.read_parquet(ROOT / config["parents"]["incumbent_ranking_predictions"]["path"]).set_index(ID)["prediction"]
    fallback = pd.read_parquet(ROOT / config["parents"]["fallback_ranking_predictions"]["path"]).set_index(ID)["prediction"]
    ids = ranking[ID].to_numpy()
    v7 = incumbent.reindex(ids).to_numpy(dtype=float)
    v5 = fallback.reindex(ids).to_numpy(dtype=float)
    if len(incumbent) != len(ids) or not np.isfinite(v7).all() or not np.isfinite(v5).all():
        raise ValueError("Frozen ranking predictions do not cover the ranking departures.")
    fitting = departures(config["refit_months"], RAW_COLUMNS + [TARGET])
    parameters = fit_parameters(fitting, config)
    values, counts = apply_rules(ranking, v7, v5, parameters, rules)
    predictions = pd.DataFrame({ID: ids, "prediction": values})
    # Every untouched row must reproduce the uploaded v7 integer exactly.
    uploaded = pd.read_parquet(ROOT / "runs/submissions/vigorous-whistle_v7.parquet").set_index(ID)[TARGET].reindex(ids).to_numpy()
    untouched = values == v7
    if not np.array_equal(np.rint(values[untouched]).astype(np.int32), uploaded[untouched]):
        raise ValueError("Untouched rows differ from the uploaded v7 export.")
    context = row_context(ranking)
    changed = ~untouched
    detail = pd.DataFrame({ID: ids[changed], "airport": ranking.loc[changed, "ADEP_mvt"].to_numpy(), "v7": v7[changed], "v8": values[changed],
                           "proxy_s": context.loc[changed, "proxy_s"].to_numpy(), "schedule_interval_s": context.loc[changed, "schedule_interval_s"].to_numpy()})
    records = lambda frame: [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()} for row in frame.to_dict(orient="records")]
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    provenance = {
        "policy": POLICY, "experiment_id": config["experiment_id"], "config_sha256": object_hash(config),
        "selection_sha256": sha256_file(ROOT / args.selection), "selected_candidate": selection["selected_candidate"],
        "rules": rules, "parameters": parameters, "row_counts": counts, "parents": config["parents"],
        "incumbent": config["incumbent"], "model_code_sha256": selection["model_code"]["sha256"], "audit_used": False,
        "refit_months": config["refit_months"], "native_fit_count": 0,
    }
    receipt = export_submission(predictions, ROOT / "data/raw/submitting.parquet", output, provenance)
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt,
              "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(ids), "parameters": parameters, "row_counts": counts,
              "untouched_rows_match_v7_export": True, "changed_rows": records(detail.sort_values("v8", ascending=False).round(3)),
              "changed_by_airport": {k: int(v) for k, v in detail.groupby("airport").size().items()},
              "hedged_rows": records(detail.loc[detail.schedule_interval_s.gt(parameters["schedule_interval_threshold_seconds"]) & detail.proxy_s.isna()]
                                     .sort_values("schedule_interval_s")[[ID, "schedule_interval_s", "v7", "v8"]].round(1))}
    write_json(ROOT / args.report, report)
    print({"output": report["output"], "sha256": receipt["sha256"], "rows": receipt["rows"], "row_counts": counts,
           "minimum": receipt["minimum_prediction_seconds"], "maximum": receipt["maximum_prediction_seconds"],
           "parameters": {k: parameters[k] for k in ["floor_seconds", "one_day_share", "one_day_taxi_median_seconds", "long_schedule_group_rows"]}})


if __name__ == "__main__":
    main()
