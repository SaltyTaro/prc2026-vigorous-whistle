"""Independent verification of a round-A adsb_ctx export (configs/adsb_ctx_v2.json) against its declaration, selection and saved networks.

As tools/verify_adsb_ctx.py, generalised to several context ensembles and a v27 incumbent. Checks, without the export tool's blend code path:
1. every seed of every declared context ensemble: its refit12 record carries the declared variant, code, configuration and seed, the saved
   model hash and the declared token content; all seeds one ADS-B ranking table and one extra-feature list; the saved model reloads and
   re-predicts a random sample of ranking rows (tabular inputs from the exact cached ADS-B table through tools/verify_adsb_nn.ranking_sample,
   context from the declared ranking token file) within --tolerance seconds of its saved ranking predictions; every ensemble column of the
   export detail (the context ensembles and the inherited ens_nn / ens_new / ens_nnl) is the plain mean of its saved seeds;
2. the detail's v9, residual and committee equal v27's recorded components, and its v27 column the recorded v27;
3. the candidate re-derived by plain arithmetic -- non-LIRF rows max(0.3 x (v9 + residual) + 0.7 x ((1 - s) x committee + s x sum_e w_e x ens_e),
   -12) with the candidate's network weights w and share s; every LIRF row the uploaded v27 -- equals the exported integers after rounding
   and the detail to 1e-6 s;
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
from prc2026.features import ID, TARGET  # noqa: E402
from build_ctx_tokens import content_hash  # noqa: E402
from verify_adsb_nn import ranking_sample  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_ctx_v2.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_ctx/dev2/selection.json"))
    parser.add_argument("--export", type=Path, default=Path("docs/adsb_ctx/export_v2.json"))
    parser.add_argument("--incumbent", type=Path, default=Path("docs/adsb_ctx/export.json"), help="v27's export record")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--tolerance", type=float, default=0.01, help="seconds; Linux-saved float32 networks reloaded on Windows differ by up to ~2e-3 s (docs/adsb_nn)")
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_ctx2/inputs/adsb_features_arcw0"), help="a byte copy of arcw0's runs/adsb_nn/features")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    from prc2026_ctx.tokens import TokenStore
    from prc2026_ctx.train import Context, load, predict
    decl = read_json(ROOT / args.declaration)
    selection = read_json(ROOT / args.selection)
    if selection["declaration_sha256"] != object_hash(decl) or not selection["selected_candidate"]:
        raise ValueError("The selection does not belong to this declaration or selected nothing.")
    name = selection["selected_candidate"]
    candidate = decl["candidates"][name]
    weights = {k: float(v) for k, v in candidate["network"].items()}
    if not set(weights) <= set(decl["networks"]) or abs(sum(weights.values()) - 1) > 1e-12:
        raise ValueError("This verifier re-derives a candidate whose network table is a convex mix of declared context ensembles.")
    share = float(candidate["network_share"])
    export = read_json(ROOT / args.export)
    if export["selected_candidate"] != name or sha256_file(ROOT / export["ranking_values"]) != export["ranking_values_sha256"]:
        raise ValueError("The export record does not match the selection or its detail changed.")
    detail = pd.read_parquet(ROOT / export["ranking_values"])
    ids = detail[ID].to_numpy()
    report = {"created_at_utc": utc_now(), "version": args.version, "selected_candidate": name, "candidate": candidate, "learners": {}, "ensembles": {}}

    # 1. the context networks
    records = {}
    for ensemble, net in decl["networks"].items():
        records[ensemble] = {leaf: read_json(ROOT / leaf / "refit12" / "run.json") for leaf in net["learners"]}
        for leaf, record in records[ensemble].items():
            run = ROOT / leaf / "refit12"
            if (record["fold"] != "refit12" or record["variant_sha256"] != net["variant_sha256"] or record["inputs"]["code"]["sha256"] != net["code"]["sha256"]
                    or record["config_sha256"] != net["config_sha256"] or record["seed"] not in net["seeds"] or record["model_sha256"] != sha256_file(run / "model.pt")
                    or record.get("audit_used") or record["inputs"]["tokens"]["files"] != net["tokens_manifest_content"]):
                raise ValueError(f"{leaf}: refit12 record does not match the declaration.")
        if sorted(r["seed"] for r in records[ensemble].values()) != sorted(net["seeds"]):
            raise ValueError(f"The {ensemble} refits do not cover the declared seeds exactly once.")
    every = [r for recs in records.values() for r in recs.values()]
    caches = {r["inputs"]["adsb_cache_sha256"]["ranking.parquet"] for r in every}
    extras = {tuple(r["inputs"]["extra_features"]) for r in every}
    manifests = {object_hash(net["tokens_manifest_content"]) for net in decl["networks"].values()}
    if len(caches) != 1 or len(extras) != 1 or len(manifests) != 1:
        raise ValueError("The context seeds used different ADS-B ranking tables, extra features or token files.")
    report["adsb_ranking_table_sha256"] = caches.pop()
    token_file = ROOT / "data/ctx/tokens_v1" / "ranking.npz"
    report["ranking_tokens_content_sha256"] = content_hash(token_file)
    if any(report["ranking_tokens_content_sha256"] != net["tokens_manifest_content"]["ranking"] for net in decl["networks"].values()):
        raise ValueError("The ranking token file is not the one the declaration records.")
    frame = ranking_sample(ids, args.sample, 20260926, list(extras.pop()), ROOT / args.adsb_cache, report["adsb_ranking_table_sha256"])
    store = TokenStore(ROOT / "data/ctx/tokens_v1", ["ranking"])
    ensembles = {}
    for ensemble, recs in records.items():
        seeds = []
        for leaf, record in recs.items():
            run = ROOT / leaf / "refit12"
            saved = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["adsb_full"]
            seeds.append(saved.reindex(ids).to_numpy(dtype=float))
            model, pre, settings = load(run)
            x_num, x_cat = pre.transform(frame)
            values = predict(model, pre, x_num, x_cat, Context.build(store, *store.lookup(frame[ID].to_numpy()), settings))
            delta = np.abs(values - saved.reindex(frame[ID]).to_numpy(dtype=float))
            report["learners"][leaf] = {"ensemble": ensemble, "seed": record["seed"], "sample_rows": int(len(frame)), "max_abs_difference_seconds": float(delta.max())}
            print(leaf, report["learners"][leaf], flush=True)
        ensembles[ensemble] = seeds
    worst = max(r["max_abs_difference_seconds"] for r in report["learners"].values())
    report["reload_tolerance_seconds"], report["reload_max_abs_difference_seconds"] = args.tolerance, worst
    if worst > args.tolerance:
        raise ValueError(f"A reloaded network differs from its saved predictions by {worst} s (tolerance {args.tolerance} s).")
    rnn = read_json(ROOT / "configs/adsb_rnn_v1.json")
    for ensemble, spec in rnn["ensembles"].items():
        ensembles[ensemble] = [pd.read_parquet(ROOT / leaf / "refit12" / "ranking_predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
                               for leaf in spec["learners"]]
    for ensemble, columns in ensembles.items():
        if ensemble not in detail:
            raise ValueError(f"The export detail has no {ensemble} column.")
        report["ensembles"][ensemble] = {"seeds": len(columns), "max_abs_difference_to_detail": float(np.max(np.abs(np.mean(columns, axis=0) - detail[ensemble].to_numpy(dtype=float))))}
        if report["ensembles"][ensemble]["max_abs_difference_to_detail"] > 1e-9:
            raise ValueError(f"The export's {ensemble} column is not the plain mean of its saved seeds.")

    # 2. components
    incumbent = read_json(ROOT / args.incumbent)
    if sha256_file(ROOT / incumbent["ranking_values"]) != incumbent["ranking_values_sha256"] or sha256_file(ROOT / incumbent["output"]) != incumbent["receipt"]["sha256"]:
        raise ValueError("The recorded v27 components or the uploaded v27 file changed.")
    recorded = pd.read_parquet(ROOT / incumbent["ranking_values"]).set_index(ID).reindex(ids)
    for column in ["v9", "residual", "committee"]:
        if not np.array_equal(detail[column].to_numpy(dtype=float), recorded[column].to_numpy(dtype=float)):
            raise ValueError(f"The export's {column} column differs from v27's.")
    report["rebuilt_v27_max_abs_difference_to_recorded"] = float(np.max(np.abs(detail["v27"].to_numpy(dtype=float) - recorded["v27"].to_numpy(dtype=float))))
    if report["rebuilt_v27_max_abs_difference_to_recorded"] > 1e-6:
        raise ValueError("The export's rebuilt v27 differs from the recorded v27.")
    committee_file = ROOT / rnn["extra_feature_tables"][rnn["committee_table"]]["ranking"]
    reference = pd.read_parquet(committee_file).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
    if not np.array_equal(detail["committee"].to_numpy(dtype=float), reference):
        raise ValueError("The export's committee column differs from the twelve-month committee table.")

    # 3. plain arithmetic
    uploaded = pd.read_parquet(ROOT / incumbent["output"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    lirf = detail["airport"].eq("LIRF").to_numpy()
    network = np.sum([w * np.mean(ensembles[e], axis=0) for e, w in weights.items()], axis=0)
    mix = (1 - share) * detail["committee"].to_numpy(dtype=float) + share * network
    plain = np.maximum(0.3 * (detail["v9"].to_numpy(dtype=float) + detail["residual"].to_numpy(dtype=float)) + 0.7 * mix, -12.0)
    out = pq.read_table(ROOT / export["output"]).to_pandas().set_index(ID)[TARGET].reindex(ids).to_numpy()
    report["non_lirf_rows_equal_after_rounding"] = bool(np.array_equal(np.rint(plain[~lirf]).astype(np.int32), out[~lirf]))
    report["non_lirf_max_abs_difference_to_detail"] = float(np.max(np.abs(plain[~lirf] - detail[f"v{args.version}"].to_numpy(dtype=float)[~lirf])))
    report["lirf_rows_equal_uploaded_v27"] = bool(np.array_equal(out[lirf], uploaded[lirf]))
    report["rows"] = {"non_lirf": int((~lirf).sum()), "lirf": int(lirf.sum())}
    if not (report["non_lirf_rows_equal_after_rounding"] and report["lirf_rows_equal_uploaded_v27"]) or report["non_lirf_max_abs_difference_to_detail"] > 1e-6:
        raise ValueError(f"The exported values do not follow the declared arithmetic: {report}")

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
    print({k: report[k] for k in ["passed", "reload_max_abs_difference_seconds", "non_lirf_rows_equal_after_rounding", "lirf_rows_equal_uploaded_v27", "rows"]})


if __name__ == "__main__":
    main()
