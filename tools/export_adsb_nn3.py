"""Export the adsb_nn3 selected candidate as the next versioned submission: v25 with the selected LIRF network and weights.

Steps (declared in configs/adsb_nn3_v1.json, "ship"); a copy of tools/export_adsb_nn2.py with v25 as the reference:
1. Every ensemble's ranking table is the mean of its declared seeds' twelve-month refits; every refit12 record must carry the ensemble's
   declared variant, code, configuration and seed, and its saved model the recorded hash.
2. The v25 ranking values are rebuilt from their recorded components (runs/adsb_nn2/export_v25/ranking_values.parquet, hash-checked
   against docs/adsb_nn2/export.json); every ensemble column recorded for v25 must be reproduced; the control must equal the recorded
   v25 values to 1e-6 s and the uploaded v25 integers exactly.
3. The selected candidate's LIRF network and weights are applied with tools/run_adsb_nn2_dev.py's own functions; floor; nearest-integer
   export through the frozen submission contract. Every row outside the two LIRF blend scopes must equal the uploaded v25.
Nothing is uploaded here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
from export_adsb_v11 import ranking_table  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_v11 import SCOPES, candidate_values, check_parents  # noqa: E402
from run_lirf_mixture import MOVEMENT_COLUMNS  # noqa: E402

RANKING_MONTHS = ["2026-01", "2026-07"]
SCOPE_ORDER = ["matched_no_lirf", "all_no_lirf", "lirf_nonpop_matched", "lirf_nonpop_unmatched"]


def ensemble_table(spec: dict, ids: np.ndarray) -> tuple[np.ndarray, dict]:
    columns, records = [], {}
    for learner in spec["learners"]:
        run = ROOT / learner / "refit12"
        record = read_json(run / "run.json")
        if (record["fold"] != "refit12" or record["variant_sha256"] != spec["variant_sha256"] or record["inputs"]["code"]["sha256"] != spec["code_sha256"]
                or record["config_sha256"] != spec["config_sha256"] or record["seed"] not in spec["seeds"] or record["model_sha256"] != sha256_file(run / "model.pt")):
            raise ValueError(f"{learner}: the twelve-month refit does not match its declared network or its saved model.")
        values = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{learner}: the twelve-month refit does not cover every ranking row.")
        columns.append(values)
        records[learner] = {"seed": record["seed"], "fitted_rows": record["fitted_rows"], "epochs": record["epochs"], "model_sha256": record["model_sha256"],
                            "run_sha256": sha256_file(run / "run.json"), "predictions_sha256": sha256_file(run / "ranking_predictions.parquet"),
                            "adsb_ranking_table_sha256": record["inputs"]["adsb_cache_sha256"]["ranking.parquet"]}
    if sorted(r["seed"] for r in records.values()) != sorted(spec["seeds"]):
        raise ValueError("The twelve-month refits do not cover the declared seeds exactly once.")
    return np.mean(columns, axis=0), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_nn3_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_nn3/dev/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_nn3/export.json"))
    args = parser.parse_args()
    started = time.time()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    config_sha256 = object_hash(config)
    selection = read_json(ROOT / args.selection)
    if selection["config_sha256"] != config_sha256 or not selection["selected_candidate"]:
        raise ValueError("The selection does not belong to this declaration or selected nothing.")
    name, control_name = selection["selected_candidate"], config["control_candidate"]
    if not selection["candidates"][name]["eligible"] or name == control_name:
        raise ValueError("The selected candidate is not eligible.")
    candidate = config["candidates"][name]
    v25_record = read_json(ROOT / config["parents"]["v25_export_record"]["path"])
    if sha256_file(ROOT / v25_record["ranking_values"]) != v25_record["ranking_values_sha256"]:
        raise ValueError("The recorded v25 ranking values changed.")
    recorded = pd.read_parquet(ROOT / v25_record["ranking_values"])
    ids = recorded[ID].to_numpy()
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    if not np.array_equal(departures[ID].to_numpy(), ids):
        raise ValueError("The recorded v25 values are not in the ranking departure order.")
    v9, residual, v25_recorded = (recorded[c].to_numpy(dtype=float) for c in ["v9", "residual", "v25"])
    runtime = {**config, "matches_dir": config["matches_dirs"]["v11"], "residual_model": {"neighbour_features": False},
               "extra_feature_tables": {k: v for k, v in config["extra_feature_tables"].items() if k not in config["ensembles"]}}
    stands = pd.read_parquet(ROOT / config["matches_dirs"]["v11"] / "stands.parquet")
    table, _ = ranking_table(runtime, departures, stands, v9)
    if not np.array_equal(table[ID].to_numpy(), ids):
        raise ValueError("Ranking table order differs from the departures.")
    floor = float(pd.read_parquet(ROOT / config["cross_fit"]["scratch"] / "2025-12.parquet", columns=["floor"])["floor"].iloc[0])
    if floor != -12.0:
        raise ValueError("Unexpected floor.")
    committee = table[config["committee_table"]].to_numpy(dtype=float)
    if not np.array_equal(committee, recorded["committee"].to_numpy(dtype=float)):
        raise ValueError("The committee table differs from v25's.")
    learners = {}
    for ensemble, spec in config["ensembles"].items():
        table[ensemble], learners[ensemble] = ensemble_table(spec, ids)
    for ensemble in [e for e in config["ensembles"] if e in recorded.columns]:
        if float(np.max(np.abs(table[ensemble].to_numpy() - recorded[ensemble].to_numpy(dtype=float)))) > 1e-9:
            raise ValueError(f"The {ensemble} mean differs from v25's recorded one.")
    uploaded = pd.read_parquet(ROOT / config["parents"]["v25_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    values_by = {}
    for label, cand in [(control_name, config["candidates"][control_name]), (name, candidate)]:
        table[f"committee_mix:{label}"] = mixed(committee, weighted(table, cand["network"]), float(config["network_share"]))
        values_by[label], _ = candidate_values(table, v9, residual, spec_for(config, f"committee_mix:{label}", lirf_blends(config, table, cand, label)), floor)
    control, values = values_by[control_name], values_by[name]
    if float(np.max(np.abs(control - v25_recorded))) > 1e-6 or not np.array_equal(np.rint(control).astype(np.int32), uploaded):
        raise ValueError("The rebuilt control does not reproduce v25.")
    if candidate["network"] != config["candidates"][control_name]["network"]:
        raise ValueError("adsb_nn3 candidates keep v25's committee pool.")
    exported = np.rint(values).astype(np.int32)
    scope = np.full(len(ids), "fixed", dtype=object)
    for s in SCOPE_ORDER:
        scope[SCOPES[s](table).to_numpy() & (scope == "fixed")] = s
    may_change = np.isin(scope, [s for s in SCOPE_ORDER if s.startswith("lirf")])
    if not np.array_equal(exported[~may_change], uploaded[~may_change]):
        raise ValueError("Rows the candidate cannot change differ from the uploaded v25.")
    run_dir = ROOT / "runs/adsb_nn3" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    row_month = departures[TIME].dt.strftime("%Y-%m").to_numpy()
    detail = pd.DataFrame({ID: ids, "airport": table["airport"].to_numpy(), "month": row_month, "scope": scope, "v9": v9, "residual": residual,
                           "committee": committee, "lirf_tree": table[config["lirf_table"]].to_numpy(dtype=float),
                           **{e: table[e].to_numpy(dtype=float) for e in config["ensembles"]},
                           "committee_mix": table[f"committee_mix:{name}"].to_numpy(), "v25": control, f"v{args.version}": values})
    detail.to_parquet(run_dir / "ranking_values.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": config["experiment_id"], "config_sha256": config_sha256, "selection_sha256": sha256_file(ROOT / args.selection),
                  "selected_candidate": name, "candidate": candidate, "ensembles": {e: {k: config["ensembles"][e][k] for k in ["variant", "variant_sha256", "code_sha256", "config_sha256", "seeds"]} for e in config["ensembles"]},
                  "learners": learners, "recipe": "v25 with the selected LIRF network and weights", "incumbent": config["incumbent"],
                  "external_data": config["external_data"], "parents": config["parents"], "audit_used": False, "floor_seconds": floor}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    delta = exported.astype(float) - uploaded
    airports = table["airport"].to_numpy()
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(pd.Series(ids)),
              "selected_candidate": name, "candidate": candidate, "learners": learners, "control_reproduces_v25": True, "rows_outside_lirf_scopes_unchanged": True,
              "rows_by_scope": {s: int((scope == s).sum()) for s in [*SCOPE_ORDER, "fixed"]},
              "ranking_values": (run_dir / "ranking_values.parquet").relative_to(ROOT).as_posix(), "ranking_values_sha256": sha256_file(run_dir / "ranking_values.parquet"),
              "versus_v25": {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds": float(np.sqrt(np.mean(delta ** 2))), "mean_change_seconds": float(delta.mean()),
                             "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(delta), q)) for q in [0.5, 0.9, 0.99, 0.999]},
                             "max_absolute_change_seconds": float(np.abs(delta).max()),
                             "rms_change_by_airport": {a: float(np.sqrt(np.mean(delta[airports == a] ** 2))) for a in sorted(set(airports))},
                             "rms_change_by_month": {m: float(np.sqrt(np.mean(delta[row_month == m] ** 2))) for m in RANKING_MONTHS},
                             "rms_change_by_scope": {s: float(np.sqrt(np.mean(delta[scope == s] ** 2))) for s in [*SCOPE_ORDER, "fixed"] if (scope == s).any()}},
              "ensemble_prediction_quantiles": {e: {str(q): float(np.quantile(table[e], q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]} for e in config["ensembles"]},
              "seconds": round(time.time() - started, 1)}
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["output", "selected_candidate", "rows_by_scope"]}, report["versus_v25"]["changed_rows"], round(report["versus_v25"]["rmse_of_change_seconds"], 3))


if __name__ == "__main__":
    main()
