"""Export the adsb_nn selected candidate as the next versioned submission: v23 with only the committee table mixed with the network.

Steps (declared in configs/adsb_nn_v1.json, "ship"):
1. The network's ranking table is the mean of the declared seeds' twelve-month refits (<learner>/refit12/ranking_predictions.parquet);
   every refit12 record must carry the declared variant, code and seed, and its saved model must match the recorded hash.
2. The v23 ranking values are rebuilt from their recorded components (runs/adsb_v25/export_v23/ranking_values.parquet: v9 and the
   residual), the label-free ranking scope table (tools/export_adsb_v11.ranking_table on the v11 match directory) and the v23
   twelve-month committee and LIRF tables. With the committee table unchanged the rebuild must equal the recorded v23 values to
   1e-6 s and the uploaded v23 integers exactly.
3. The committee table becomes (1 - a) * committee + a * network for the selected a; blends and floor as v23; nearest-integer export
   through the frozen submission contract. Every LIRF row must equal the uploaded v23.
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
from run_adsb_v11 import candidate_values, check_parents  # noqa: E402
from run_adsb_nn_dev import mixed, spec_for  # noqa: E402
from run_lirf_mixture import MOVEMENT_COLUMNS  # noqa: E402

RANKING_MONTHS = ["2026-01", "2026-07"]


def network_table(config: dict, ids: np.ndarray) -> tuple[np.ndarray, dict]:
    spec = config["network"]
    columns, records = [], {}
    for learner in spec["learners"]:
        run = ROOT / learner / "refit12"
        record = read_json(run / "run.json")
        if (record["fold"] != "refit12" or record["variant_sha256"] != spec["variant_sha256"] or record["inputs"]["code"]["sha256"] != spec["code_sha256"]
                or record["config_sha256"] != spec["config_sha256"] or record["seed"] not in spec["seeds"] or record["model_sha256"] != sha256_file(run / "model.pt")):
            raise ValueError(f"{learner}: the twelve-month refit does not match the declared network or its saved model.")
        values = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{learner}: the twelve-month refit does not cover every ranking row.")
        columns.append(values)
        records[learner] = {"seed": record["seed"], "fitted_rows": record["fitted_rows"], "epochs": record["epochs"], "model_sha256": record["model_sha256"],
                            "run_sha256": sha256_file(run / "run.json"), "predictions_sha256": sha256_file(run / "ranking_predictions.parquet")}
    if sorted(r["seed"] for r in records.values()) != sorted(spec["seeds"]):
        raise ValueError("The twelve-month refits do not cover the declared seeds exactly once.")
    return np.mean(columns, axis=0), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_nn_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_nn/dev/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_nn/export.json"))
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
    name = selection["selected_candidate"]
    if not selection["candidates"][name]["eligible"] or name == config["control_candidate"]:
        raise ValueError("The selected candidate is not eligible.")
    share = float(config["candidates"][name])
    recorded = pd.read_parquet(ROOT / config["parents"]["v23_ranking_values"]["path"])
    ids = recorded[ID].to_numpy()
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    if not np.array_equal(departures[ID].to_numpy(), ids):
        raise ValueError("The recorded v23 values are not in the ranking departure order.")
    v9, residual, v23_recorded = (recorded[c].to_numpy(dtype=float) for c in ["v9", "residual", "v23"])
    # the scope table with the v23 twelve-month committee and LIRF tables attached (the network table is added below)
    runtime = {**config, "matches_dir": config["matches_dirs"]["v11"], "residual_model": {"neighbour_features": False},
               "extra_feature_tables": {k: v for k, v in config["extra_feature_tables"].items() if k != config["network_table"]}}
    stands = pd.read_parquet(ROOT / config["matches_dirs"]["v11"] / "stands.parquet")
    table, _ = ranking_table(runtime, departures, stands, v9)
    if not np.array_equal(table[ID].to_numpy(), ids):
        raise ValueError("Ranking table order differs from the departures.")
    floor = float(pd.read_parquet(ROOT / config["cross_fit"]["scratch"] / "2025-12.parquet", columns=["floor"])["floor"].iloc[0])
    if floor != -12.0:
        raise ValueError("Unexpected floor.")
    committee = table[config["committee_table"]].to_numpy(dtype=float)
    network, learners = network_table(config, ids)
    table["committee_mix:control"] = committee
    control, _ = candidate_values(table, v9, residual, spec_for(config, "committee_mix:control"), floor)
    uploaded = pd.read_parquet(ROOT / config["parents"]["v23_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if float(np.max(np.abs(control - v23_recorded))) > 1e-6 or not np.array_equal(np.rint(control).astype(np.int32), uploaded):
        raise ValueError("The rebuilt control does not reproduce v23.")
    table[f"committee_mix:{name}"] = mixed(committee, network, share)
    values, touched = candidate_values(table, v9, residual, spec_for(config, f"committee_mix:{name}"), floor)
    exported = np.rint(values).astype(np.int32)
    lirf = table["airport"].eq("LIRF").to_numpy()
    if not np.array_equal(exported[lirf], uploaded[lirf]):
        raise ValueError("LIRF rows differ from the uploaded v23.")
    run_dir = ROOT / "runs/adsb_nn" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    row_month = departures[TIME].dt.strftime("%Y-%m").to_numpy()
    detail = pd.DataFrame({ID: ids, "airport": table["airport"].to_numpy(), "month": row_month, "v9": v9, "residual": residual, "committee": committee, "network": network,
                           "committee_mix": table[f"committee_mix:{name}"].to_numpy(), "v23": control, f"v{args.version}": values})
    detail.to_parquet(run_dir / "ranking_values.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": config["experiment_id"], "config_sha256": config_sha256, "selection_sha256": sha256_file(ROOT / args.selection),
                  "selected_candidate": name, "network_share_in_committee": share, "network": {"variant": config["network"]["variant"], "variant_sha256": config["network"]["variant_sha256"],
                  "code_sha256": config["network"]["code_sha256"], "learners": learners}, "recipe": "v23 with the committee table mixed", "incumbent": config["incumbent"],
                  "external_data": config["external_data"], "parents": config["parents"], "audit_used": False, "floor_seconds": floor}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    delta = exported.astype(float) - uploaded
    airports = table["airport"].to_numpy()
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(pd.Series(ids)),
              "selected_candidate": name, "network_share_in_committee": share, "learners": learners, "control_reproduces_v23": True, "lirf_rows_unchanged": True,
              "touched_rows": int(touched.sum()), "ranking_values": (run_dir / "ranking_values.parquet").relative_to(ROOT).as_posix(),
              "ranking_values_sha256": sha256_file(run_dir / "ranking_values.parquet"),
              "versus_v23": {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds": float(np.sqrt(np.mean(delta ** 2))), "mean_change_seconds": float(delta.mean()),
                             "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(delta), q)) for q in [0.5, 0.9, 0.99, 0.999]}, "max_absolute_change_seconds": float(np.abs(delta).max()),
                             "rms_change_by_airport": {a: float(np.sqrt(np.mean(delta[airports == a] ** 2))) for a in sorted(set(airports))},
                             "rms_change_by_month": {m: float(np.sqrt(np.mean(delta[row_month == m] ** 2))) for m in RANKING_MONTHS}},
              "network_prediction_quantiles": {str(q): float(np.quantile(network, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]},
              "seconds": round(time.time() - started, 1)}
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["output", "selected_candidate", "network_share_in_committee", "touched_rows"]}, report["versus_v23"]["changed_rows"], round(report["versus_v23"]["rmse_of_change_seconds"], 3))


if __name__ == "__main__":
    main()
