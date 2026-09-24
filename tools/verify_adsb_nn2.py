"""Independent verification of an adsb_nn2 export against its declaration, selection and saved networks.

Checks, without the export tool's blend code path:
1. every seed of every ensemble the selected candidate uses (and v24's sixteen): its refit12 record carries the ensemble's declared variant,
   code, configuration and seed and the saved model hash; the saved model reloads (prc2026_nn.train.load) and re-predicts a random sample
   of ranking rows, rebuilt through tools/run_adsb_nn.py's data path from the exact cached ADS-B table the networks used, within
   --tolerance seconds of its saved ranking predictions; each ensemble column of the export detail is the plain mean of its saved seeds;
2. the detail's committee and LIRF tree columns equal the v23 twelve-month tables; v9 and residual equal v24's recorded components;
   the detail's scope labels equal masks recomputed here from the ranking scope table with literal copies of the scope definitions;
3. the candidate re-derived by plain arithmetic -- non-LIRF rows 0.3 x (v9 + residual) + 0.7 x (0.5 x committee + 0.5 x pool); rows of
   the two LIRF scopes v9 x w_v9 + tree x w_tree + nnl-mean x w_net; floor -12; every other LIRF row the uploaded v24 -- equals the
   exported integers after rounding and the detail to 1e-6 s;
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
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TARGET  # noqa: E402
from verify_adsb_nn import ranking_sample  # noqa: E402

LIRF_SCOPES = {  # literal copies of tools/run_adsb_v11.py / run_adsb_v24.py
    "lirf_nonpop_matched": lambda t: (t["airport"].eq("LIRF") & ~t["population"].astype(bool) & t["adsb_matched"].eq(1)
                                      & t["proxy_s"].abs().lt(21600) & t["schedule_interval_s"].abs().lt(21600)),
    "lirf_nonpop_unmatched": lambda t: (t["airport"].eq("LIRF") & ~t["population"].astype(bool) & t["adsb_matched"].ne(1)
                                        & t["proxy_s"].abs().lt(21600) & t["schedule_interval_s"].abs().lt(21600)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_nn2_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_nn2/dev/selection.json"))
    parser.add_argument("--export", type=Path, default=Path("docs/adsb_nn2/export.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--tolerance", type=float, default=0.01, help="seconds; Linux-saved float32 networks reloaded on Windows differ by up to ~2e-3 s (docs/adsb_nn)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_nn/features_vm"))
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
    if export["selected_candidate"] != name or sha256_file(ROOT / export["ranking_values"]) != export["ranking_values_sha256"]:
        raise ValueError("The export record does not match the selection or its detail changed.")
    detail = pd.read_parquet(ROOT / export["ranking_values"])
    ids = detail[ID].to_numpy()
    report = {"created_at_utc": utc_now(), "version": args.version, "selected_candidate": name, "candidate": candidate, "learners": {}, "ensembles": {}}
    # 1. networks
    used = sorted(set(candidate["network"]) | set(candidate["lirf_network"] if any(w["net"] > 0 for w in candidate["lirf"].values()) else []) | {config["v24_network_table"]})
    caches = {read_json(ROOT / leaf / "refit12" / "run.json")["inputs"]["adsb_cache_sha256"]["ranking.parquet"] for e in used for leaf in config["ensembles"][e]["learners"]}
    if len(caches) != 1:
        raise ValueError("The seeds were fitted with different ADS-B ranking tables.")
    report["adsb_ranking_table_sha256"] = caches.pop()
    extra = config["ensembles"][used[0]]["variant_settings"]["features"]["extra_features"]
    if any(config["ensembles"][e]["variant_settings"]["features"]["extra_features"] != extra for e in used):
        raise ValueError("The ensembles use different extra features.")
    frame = ranking_sample(ids, args.sample, 20260924, extra, ROOT / args.adsb_cache, report["adsb_ranking_table_sha256"])
    for ensemble in used:
        spec, seeds = config["ensembles"][ensemble], []
        for learner in spec["learners"]:
            run = ROOT / learner / "refit12"
            record = read_json(run / "run.json")
            if (record["variant_sha256"] != spec["variant_sha256"] or record["inputs"]["code"]["sha256"] != spec["code_sha256"]
                    or record["config_sha256"] != spec["config_sha256"] or record["seed"] not in spec["seeds"] or record["model_sha256"] != sha256_file(run / "model.pt")):
                raise ValueError(f"{learner}: refit12 record does not match the declaration.")
            saved = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["adsb_full"]
            seeds.append(saved.reindex(ids).to_numpy(dtype=float))
            model, pre = load(run)
            x_num, x_cat = pre.transform(frame)
            if record.get("target_offset", "none") != "none":
                raise ValueError(f"{learner}: an offset target is not part of this verifier.")
            delta = np.abs(predict(model, pre, x_num, x_cat) - saved.reindex(frame[ID]).to_numpy(dtype=float))
            report["learners"][learner] = {"ensemble": ensemble, "seed": record["seed"], "sample_rows": int(len(frame)), "max_abs_difference_seconds": float(delta.max())}
        mean = np.mean(seeds, axis=0)
        report["ensembles"][ensemble] = {"seeds": len(seeds), "max_abs_difference_to_detail": float(np.max(np.abs(mean - detail[ensemble].to_numpy(dtype=float))))}
        if report["ensembles"][ensemble]["max_abs_difference_to_detail"] > 1e-9:
            raise ValueError(f"The export's {ensemble} column is not the plain mean of its saved seeds.")
    worst = max(r["max_abs_difference_seconds"] for r in report["learners"].values())
    report["reload_tolerance_seconds"], report["reload_max_abs_difference_seconds"] = args.tolerance, worst
    print("reload worst", worst, flush=True)
    if worst > args.tolerance:
        raise ValueError(f"A reloaded network differs from its saved predictions by {worst} s (tolerance {args.tolerance} s).")
    # 2. components and scopes
    v24 = read_json(ROOT / config["parents"]["v24_export_record"]["path"])
    recorded = pd.read_parquet(ROOT / v24["ranking_values"]).set_index(ID).reindex(ids)
    for column, source in [("committee", config["committee_table"]), ("lirf_tree", config["lirf_table"])]:
        reference = pd.read_parquet(ROOT / config["extra_feature_tables"][source]["ranking"]).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
        if not np.array_equal(detail[column].to_numpy(dtype=float), reference):
            raise ValueError(f"The export's {column} column differs from {source}.")
    for column in ["v9", "residual"]:
        if not np.array_equal(detail[column].to_numpy(dtype=float), recorded[column].to_numpy(dtype=float)):
            raise ValueError(f"The export's {column} column differs from v24's.")
    from export_adsb_v11 import ranking_table
    from prc2026_adsb.features import DEPARTURE_COLUMNS
    from run_lirf_mixture import MOVEMENT_COLUMNS
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    runtime = {**config, "matches_dir": config["matches_dirs"]["v11"], "residual_model": {"neighbour_features": False}, "extra_feature_tables": {}}
    scope_table, _ = ranking_table(runtime, departures, pd.read_parquet(ROOT / config["matches_dirs"]["v11"] / "stands.parquet"), detail["v9"].to_numpy(dtype=float))
    if not np.array_equal(scope_table[ID].to_numpy(), ids):
        raise ValueError("Scope table order differs.")
    scope = detail["scope"].to_numpy()
    lirf = detail["airport"].eq("LIRF").to_numpy()
    for s, rule in LIRF_SCOPES.items():
        if not np.array_equal(rule(scope_table).to_numpy(), scope == s):
            raise ValueError(f"The export's {s} labels differ from the recomputed scope.")
    # 3. plain arithmetic
    uploaded = pd.read_parquet(ROOT / config["parents"]["v24_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    pool = sum(w * detail[e].to_numpy(dtype=float) for e, w in candidate["network"].items())
    mix = 0.5 * detail["committee"].to_numpy(dtype=float) + 0.5 * pool
    plain = (1 - 0.7) * (detail["v9"].to_numpy(dtype=float) + detail["residual"].to_numpy(dtype=float)) + 0.7 * mix
    for s, w in candidate["lirf"].items():
        k = scope == s
        net = sum(x * detail[e].to_numpy(dtype=float)[k] for e, x in candidate["lirf_network"].items()) if w["net"] > 0 else 0.0
        plain[k] = w["v9"] * detail["v9"].to_numpy(dtype=float)[k] + w["tree"] * detail["lirf_tree"].to_numpy(dtype=float)[k] + w["net"] * net
    plain = np.maximum(plain, -12.0)
    other_lirf = lirf & ~np.isin(scope, list(LIRF_SCOPES))
    out = pq.read_table(ROOT / export["output"]).to_pandas().set_index(ID)[TARGET].reindex(ids).to_numpy()
    derived = ~other_lirf
    report["derived_rows_equal_after_rounding"] = bool(np.array_equal(np.rint(plain[derived]).astype(np.int32), out[derived]))
    report["derived_max_abs_difference_to_detail"] = float(np.max(np.abs(plain[derived] - detail[f"v{args.version}"].to_numpy(dtype=float)[derived])))
    report["other_lirf_rows_equal_uploaded_v24"] = bool(np.array_equal(out[other_lirf], uploaded[other_lirf]))
    report["rows"] = {"derived": int(derived.sum()), "other_lirf": int(other_lirf.sum())}
    if not (report["derived_rows_equal_after_rounding"] and report["other_lirf_rows_equal_uploaded_v24"]) or report["derived_max_abs_difference_to_detail"] > 1e-6:
        raise ValueError("The exported values do not follow the declared arithmetic.")
    # 4. file contract
    template = pq.read_table(ROOT / "data/raw/submitting.parquet")
    table = pq.read_table(ROOT / export["output"])
    report["file"] = {"sha256": sha256_file(ROOT / export["output"]), "receipt_sha256": export["receipt"]["sha256"], "rows": table.num_rows,
                      "ids_equal_template_order": bool(np.array_equal(table[ID].to_numpy(), template[ID].to_numpy())), "types": {f.name: str(f.type) for f in table.schema}}
    if report["file"]["sha256"] != report["file"]["receipt_sha256"] or not report["file"]["ids_equal_template_order"] or report["file"]["types"] != {ID: "double", TARGET: "int32"}:
        raise ValueError("The submission file breaks the contract.")
    report["passed"] = True
    report["environment"] = environment(ROOT)
    write_json(ROOT / args.output, report)
    print({k: report[k] for k in ["passed", "reload_max_abs_difference_seconds", "derived_rows_equal_after_rounding", "other_lirf_rows_equal_uploaded_v24", "rows"]})


if __name__ == "__main__":
    main()
