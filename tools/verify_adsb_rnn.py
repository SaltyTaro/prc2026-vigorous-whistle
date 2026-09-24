"""Independent verification of an adsb_rnn export against its declaration, selection and saved networks.

Checks, without the export tool's code path:
1. every declared seed's refit12 record carries the declared variant, code, configuration and seed, the saved ranking table's hash and
   the saved model's hash; the model reloads (prc2026_nn.train.load) and re-predicts EVERY in-scope ranking row of the saved ranking
   table, read here directly, within --tolerance seconds of its saved predictions (Linux-saved float32 networks reloaded on Windows
   differ by up to ~2e-3 s); the detail's network column is the plain mean of the saved seeds, zero outside the scope;
2. the detail's v9, v25 residual, committee mix, scope and v25 equal v25's recorded ranking values; v25's residual equals the mean of
   the recorded adj2 and cat ranking residuals;
3. the candidate re-derived by plain arithmetic -- on matched_no_lirf rows max(0.3 x (v9 + (1 - w) x v25 residual + w x network) +
   0.7 x committee mix, -12), every other row the uploaded v25 -- equals the exported integers after rounding and the detail to 1e-6 s;
4. the submission file has the template's IDs, order and types, and its SHA-256 equals the receipt.
Writes a JSON report; raises on any failure. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_rnn_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_rnn/dev/selection.json"))
    parser.add_argument("--export", type=Path, default=Path("docs/adsb_rnn/export.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    from prc2026_nn.train import load, predict
    config = read_json(ROOT / args.config)
    selection = read_json(ROOT / args.selection)
    if selection["config_sha256"] != object_hash(config) or not selection["selected_candidate"]:
        raise ValueError("The selection does not belong to this declaration or selected nothing.")
    name = selection["selected_candidate"]
    candidate = config["candidates"][name]
    export = read_json(ROOT / args.export)
    if export["selected_candidate"] != name:
        raise ValueError("The export record is for another candidate.")
    detail = pd.read_parquet(ROOT / export["ranking_values"])
    if sha256_file(ROOT / export["ranking_values"]) != export["ranking_values_sha256"]:
        raise ValueError("The export detail changed.")
    ids = detail[ID].to_numpy()
    report = {"created_at_utc": utc_now(), "selected_candidate": name, "candidate": candidate, "export_sha256": sha256_file(ROOT / args.export)}
    # 1. networks
    network = [m for m in candidate["residual"] if m in config["residual_networks"]][0]
    w = float(candidate["residual"][network])
    spec = config["residual_networks"][network]
    receipt = read_json(ROOT / "runs/adsb_rnn/ranking_nb.json")
    table_path = ROOT / "runs/adsb_rnn/ranking_nb.parquet"
    if sha256_file(table_path) != receipt["table_sha256"] or receipt["table_sha256"] != config["parents"]["ranking_table"]["sha256"]:
        raise ValueError("The saved ranking table differs from the declared one.")
    table = pd.read_parquet(table_path)
    if not np.array_equal(table[ID].to_numpy(), ids):
        raise ValueError("The saved ranking table is not in the export's order.")
    inside = table["in_scope"].to_numpy(dtype=bool)
    rows = table.loc[inside].reset_index(drop=True)
    columns, reload = [], {}
    for learner in spec["learners"]:
        run = ROOT / learner / "refit12"
        record = read_json(run / "run.json")
        if (record["stage"] != "refit12" or record["variant_sha256"] != spec["variant_sha256"] or record["inputs"]["code"]["sha256"] != spec["code_sha256"]
                or record["config_sha256"] != spec["config_sha256"] or record["seed"] not in spec["seeds"] or record["model_sha256"] != sha256_file(run / "model.pt")
                or record["ranking_table_sha256"] != receipt["table_sha256"]):
            raise ValueError(f"{learner}: refit record does not match the declaration.")
        saved = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["residual"].reindex(ids).to_numpy(dtype=float)
        model, pre = load(run)
        x_num, x_cat = pre.transform(rows)
        again = predict(model, pre, x_num, x_cat)
        reload[learner] = float(np.max(np.abs(again - saved[inside])))
        if reload[learner] > args.tolerance or np.any(saved[~inside] != 0):
            raise ValueError(f"{learner}: the reloaded network does not reproduce its saved ranking predictions.")
        columns.append(saved)
    mean = np.mean(columns, axis=0)
    report["reload_max_abs_difference_seconds"] = max(reload.values())
    report["reload_by_learner"] = reload
    report["network_column_max_abs_difference"] = float(np.max(np.abs(mean - detail["network"].to_numpy(dtype=float))))
    if report["network_column_max_abs_difference"] > 1e-9:
        raise ValueError("The detail's network column is not the mean of the saved seeds.")
    # 2. v25 components
    v25_record = read_json(ROOT / config["parents"]["v25_export_record"]["path"])
    recorded = pd.read_parquet(ROOT / v25_record["ranking_values"]).set_index(ID).reindex(ids)
    same = {c: bool(np.array_equal(detail[c].to_numpy(), recorded[s].to_numpy())) for c, s in [("v9", "v9"), ("residual_v25", "residual"), ("committee_mix", "committee_mix"), ("v25", "v25"), ("scope", "scope")]}
    trees = pd.read_parquet(ROOT / "runs/adsb_v25/export_v23/ranking_values.parquet").set_index(ID).reindex(ids)
    same["residual_v25_is_mean_adj2_cat"] = bool(np.array_equal(detail["residual_v25"].to_numpy(dtype=float), (trees["residual:adj2"].to_numpy(dtype=float) + trees["residual:cat"].to_numpy(dtype=float)) / 2))
    same["scope_equals_saved_table"] = bool(np.array_equal(detail["scope"].eq("matched_no_lirf").to_numpy(), inside))
    report["v25_components_equal"] = same
    if not all(same.values()):
        raise ValueError("The export detail does not carry v25's recorded components.")
    # 3. plain arithmetic
    v9, r25, mix = (detail[c].to_numpy(dtype=float) for c in ["v9", "residual_v25", "committee_mix"])
    plain = np.maximum(0.3 * (v9 + (1 - w) * r25 + w * mean) + 0.7 * mix, -12.0)
    out = pq.read_table(ROOT / export["output"]).to_pandas().set_index(ID)[TARGET].reindex(ids).to_numpy()
    uploaded = pq.read_table(ROOT / config["parents"]["v25_export"]["path"]).to_pandas().set_index(ID)[TARGET].reindex(ids).to_numpy()
    report["scope_rows_equal_after_rounding"] = bool(np.array_equal(np.rint(plain[inside]).astype(np.int32), out[inside]))
    report["scope_max_abs_difference_to_detail"] = float(np.max(np.abs(plain[inside] - detail[f"v{args.version}"].to_numpy(dtype=float)[inside])))
    report["other_rows_equal_uploaded_v25"] = bool(np.array_equal(out[~inside], uploaded[~inside]))
    report["rows"] = {"in_scope": int(inside.sum()), "other": int((~inside).sum()), "changed_vs_v25": int((out != uploaded).sum())}
    if not (report["scope_rows_equal_after_rounding"] and report["other_rows_equal_uploaded_v25"]) or report["scope_max_abs_difference_to_detail"] > 1e-6:
        raise ValueError("The exported values do not follow the declared arithmetic.")
    # 4. file contract
    template = pq.read_table(ROOT / "data/raw/submitting.parquet")
    file_table = pq.read_table(ROOT / export["output"])
    report["file"] = {"sha256": sha256_file(ROOT / export["output"]), "receipt_sha256": export["receipt"]["sha256"], "rows": file_table.num_rows,
                      "ids_equal_template_order": bool(np.array_equal(file_table[ID].to_numpy(), template[ID].to_numpy())), "types": {f.name: str(f.type) for f in file_table.schema}}
    if report["file"]["sha256"] != report["file"]["receipt_sha256"] or not report["file"]["ids_equal_template_order"] or report["file"]["types"] != {ID: "double", TARGET: "int32"}:
        raise ValueError("The submission file breaks the contract.")
    report["passed"] = True
    report["environment"] = environment(ROOT)
    write_json(ROOT / args.output, report)
    print({k: report[k] for k in ["passed", "reload_max_abs_difference_seconds", "scope_rows_equal_after_rounding", "other_rows_equal_uploaded_v25", "rows"]})


if __name__ == "__main__":
    main()
