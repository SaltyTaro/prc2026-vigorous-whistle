"""Export the adsb_rnn selected candidate as the next versioned submission: v25 with the residual stage's network member added.

Steps (declared in configs/adsb_rnn_v1.json, "ship"):
1. The network's ranking column is the mean of its declared seeds' twelve-month refits (<learner>/refit12/ranking_predictions.parquet,
   tools/refit_adsb_rnn.py); every refit record must carry the declared variant, code, configuration and seed, the saved ranking table's
   hash, and its saved model the recorded hash. Its scope flag must equal v25's recorded residual scope (matched_no_lirf).
2. v25 is rebuilt from its recorded components (runs/adsb_nn2/export_v25/ranking_values.parquet, hash-checked against
   docs/adsb_nn2/export.json): v25's residual must equal the mean of the recorded adj2 and cat ranking residuals (v23's export detail);
   on the residual scope max(0.3 x (v9 + residual) + 0.7 x committee_mix, floor) must equal the recorded v25 to 1e-6 s, and the recorded
   v25 must round to the uploaded v25 exactly.
3. The selected candidate's residual -- (1 - w) x mean(adj2, cat) + w x network -- replaces v25's on the residual scope with the same
   arithmetic; every other row keeps the recorded v25 value. Nearest-integer export through the frozen submission contract; rows outside
   the residual scope must equal the uploaded v25.
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
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
from run_adsb_v11 import check_parents  # noqa: E402

RANKING_MONTHS = ["2026-01", "2026-07"]


def network_column(spec: dict, ids: np.ndarray, table_sha256: str) -> tuple[np.ndarray, np.ndarray, dict]:
    columns, scopes, records = [], [], {}
    for learner in spec["learners"]:
        run = ROOT / learner / "refit12"
        record = read_json(run / "run.json")
        if (record["stage"] != "refit12" or record["variant_sha256"] != spec["variant_sha256"] or record["inputs"]["code"]["sha256"] != spec["code_sha256"]
                or record["config_sha256"] != spec["config_sha256"] or record["seed"] not in spec["seeds"] or record["model_sha256"] != sha256_file(run / "model.pt")
                or record["ranking_table_sha256"] != table_sha256 or record["smoke"] or record["audit_used"]):
            raise ValueError(f"{learner}: the twelve-month refit does not match its declared network, the saved ranking table or its saved model.")
        frame = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID).reindex(ids)
        if frame["residual"].isna().any():
            raise ValueError(f"{learner}: the twelve-month refit does not cover every ranking row.")
        columns.append(frame["residual"].to_numpy(dtype=float))
        scopes.append(frame["in_scope"].to_numpy(dtype=bool))
        records[learner] = {"seed": record["seed"], "fitted_rows": record["fitted_rows"], "epochs": record["epochs"], "model_sha256": record["model_sha256"],
                            "run_sha256": sha256_file(run / "run.json"), "predictions_sha256": sha256_file(run / "ranking_predictions.parquet")}
    if sorted(r["seed"] for r in records.values()) != sorted(spec["seeds"]) or any(not np.array_equal(s, scopes[0]) for s in scopes):
        raise ValueError("The twelve-month refits do not cover the declared seeds exactly once on one scope.")
    return np.mean(columns, axis=0), scopes[0], records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_rnn_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_rnn/dev/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_rnn/export.json"))
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
    candidate = config["candidates"][name]
    weights = candidate["residual"]
    networks = [m for m in weights if m in config["residual_networks"]]
    if "recipe_weight" in candidate or len(networks) != 1 or abs(weights["adj2"] - weights["cat"]) > 1e-12:
        raise ValueError("The export handles one network with equal adj2/cat weights and v25's committee-mix weight.")
    network, w = networks[0], float(weights[networks[0]])
    v25_record = read_json(ROOT / config["parents"]["v25_export_record"]["path"])
    if sha256_file(ROOT / v25_record["ranking_values"]) != v25_record["ranking_values_sha256"]:
        raise ValueError("The recorded v25 ranking values changed.")
    recorded = pd.read_parquet(ROOT / v25_record["ranking_values"])
    ids = recorded[ID].to_numpy()
    v9, residual_v25, mix, v25 = (recorded[c].to_numpy(dtype=float) for c in ["v9", "residual", "committee_mix", "v25"])
    scope = recorded["scope"].to_numpy()
    inside = scope == config["residual_scope"]
    trees = pd.read_parquet(ROOT / "runs/adsb_v25/export_v23/ranking_values.parquet").set_index(ID).reindex(ids)
    if float(np.max(np.abs(residual_v25 - (trees["residual:adj2"].to_numpy(dtype=float) + trees["residual:cat"].to_numpy(dtype=float)) / 2))) > 0.0:
        raise ValueError("v25's residual is not the mean of the recorded adj2 and cat ranking residuals.")
    table_receipt = read_json(ROOT / config["parents"]["ranking_table_receipt"]["path"])
    net, net_scope, learners = network_column(config["residual_networks"][network], ids, table_receipt["table_sha256"])
    if not np.array_equal(net_scope, inside) or np.any(net[~inside] != 0) or np.any(residual_v25[~inside] != 0):
        raise ValueError("The network's scope differs from v25's residual scope.")
    floor = float(pd.read_parquet(ROOT / config["cross_fit"]["scratch"] / "2025-12.parquet", columns=["floor"])["floor"].iloc[0])
    outer = float(config["recipe_blend"]["weight"])
    if floor != -12.0 or outer != 0.7 or config["recipe_blend"]["scope"] != "all_no_lirf" or not np.isfinite(mix[inside]).all():
        raise ValueError("Unexpected floor, committee-mix weight or committee mix.")
    uploaded = pd.read_parquet(ROOT / config["parents"]["v25_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    control = np.where(inside, np.maximum((1 - outer) * (v9 + residual_v25) + outer * mix, floor), v25)
    if float(np.max(np.abs(control - v25))) > 1e-6 or not np.array_equal(np.rint(v25).astype(np.int32), uploaded):
        raise ValueError("The rebuilt control does not reproduce v25.")
    residual = np.where(inside, (1 - w) * residual_v25 + w * net, 0.0)
    values = np.where(inside, np.maximum((1 - outer) * (v9 + residual) + outer * mix, floor), v25)
    exported = np.rint(values).astype(np.int32)
    if not np.array_equal(exported[~inside], uploaded[~inside]):
        raise ValueError("Rows outside the residual scope differ from the uploaded v25.")
    run_dir = ROOT / "runs/adsb_rnn" / f"export_v{args.version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    row_month = recorded["month"].to_numpy()
    detail = pd.DataFrame({ID: ids, "airport": recorded["airport"].to_numpy(), "month": row_month, "scope": scope, "v9": v9, "residual_v25": residual_v25,
                           "network": net, "residual": residual, "committee_mix": mix, "v25": v25, f"v{args.version}": values})
    detail.to_parquet(run_dir / "ranking_values.parquet", index=False, compression="zstd")
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    provenance = {"policy": config["prediction_postprocess"], "experiment_id": config["experiment_id"], "config_sha256": config_sha256, "selection_sha256": sha256_file(ROOT / args.selection),
                  "selected_candidate": name, "candidate": candidate, "network": {k: config["residual_networks"][network][k] for k in ["variant", "variant_sha256", "code_sha256", "config_sha256", "seeds"]},
                  "learners": learners, "ranking_table_sha256": table_receipt["table_sha256"], "recipe": "v25 with the residual stage's network member",
                  "incumbent": config["incumbent"], "external_data": config["external_data"], "parents": config["parents"], "audit_used": False, "floor_seconds": floor}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    delta = exported.astype(float) - uploaded
    airports = recorded["airport"].to_numpy()
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "ranking_rows": int(len(ids)), "ranking_ids_sha256": identity_hash(pd.Series(ids)),
              "selected_candidate": name, "candidate": candidate, "network_weight": w, "learners": learners, "control_reproduces_v25": True, "rows_outside_scope_unchanged": True,
              "rows_in_scope": int(inside.sum()), "ranking_values": (run_dir / "ranking_values.parquet").relative_to(ROOT).as_posix(),
              "ranking_values_sha256": sha256_file(run_dir / "ranking_values.parquet"),
              "network_vs_v25_residual": {"correlation": float(np.corrcoef(net[inside], residual_v25[inside])[0, 1]),
                                          "network_quantiles": {str(q): float(np.quantile(net[inside], q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]},
                                          "v25_residual_quantiles": {str(q): float(np.quantile(residual_v25[inside], q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]}},
              "versus_v25": {"changed_rows": int((delta != 0).sum()), "rmse_of_change_seconds": float(np.sqrt(np.mean(delta ** 2))), "mean_change_seconds": float(delta.mean()),
                             "absolute_change_quantiles_seconds": {str(q): float(np.quantile(np.abs(delta), q)) for q in [0.5, 0.9, 0.99, 0.999]},
                             "max_absolute_change_seconds": float(np.abs(delta).max()),
                             "rms_change_by_airport": {a: float(np.sqrt(np.mean(delta[airports == a] ** 2))) for a in sorted(set(airports))},
                             "rms_change_by_month": {m: float(np.sqrt(np.mean(delta[row_month == m] ** 2))) for m in RANKING_MONTHS}},
              "seconds": round(time.time() - started, 1)}
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["output", "selected_candidate", "rows_in_scope"]}, report["versus_v25"]["changed_rows"], round(report["versus_v25"]["rmse_of_change_seconds"], 3))


if __name__ == "__main__":
    main()
