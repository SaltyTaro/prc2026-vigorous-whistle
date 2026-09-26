"""Independent verification of a round-B export (configs/adsb_ctx_lirf_v1.json on top of v28) against its declaration, selection and networks.

Checks, without the export tool's blend code path:
1. every seed of the new LIRF context ensemble: its refit12 record carries the declared variant, code, configuration and seed, the saved
   model hash and the declared token content, and v28's ADS-B ranking table; the saved model reloads and re-predicts a random sample of
   ranking rows (exact cached ADS-B table, declared ranking token file) within --tolerance seconds of its saved ranking predictions;
   every ensemble column of the export detail is the plain mean of its saved seeds;
2. the detail's v9, residual, committee, committee mix and v28 equal v28's recorded components; its LIRF tree table equals the
   twelve-month LIRF table file;
3. plain arithmetic on the two LIRF non-population scopes, value = max(w_v9 x v9 + w_tree x tree + w_net x net, -12) where the mix is
   finite (else max(v9, -12)), net the weighted mean of the named ensembles: with v25's weights and ens_nnl it must give the recorded v28
   (1e-6 s), with the selected candidate's the exported integers after rounding and the detail (1e-6 s); every other row equals the
   uploaded v28;
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

LIRF_SCOPES = ("lirf_nonpop_matched", "lirf_nonpop_unmatched")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_ctx_lirf_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_ctx/dev_lirf/selection.json"))
    parser.add_argument("--export", type=Path, default=Path("docs/adsb_ctx_lirf/export.json"))
    parser.add_argument("--incumbent", type=Path, default=Path("docs/adsb_ctx/export_v2.json"), help="v28's export record")
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
    export = read_json(ROOT / args.export)
    if export["selected_candidate"] != name or sha256_file(ROOT / export["ranking_values"]) != export["ranking_values_sha256"]:
        raise ValueError("The export record does not match the selection or its detail changed.")
    detail = pd.read_parquet(ROOT / export["ranking_values"])
    ids = detail[ID].to_numpy()
    incumbent = read_json(ROOT / args.incumbent)
    if sha256_file(ROOT / incumbent["ranking_values"]) != incumbent["ranking_values_sha256"] or sha256_file(ROOT / incumbent["output"]) != incumbent["receipt"]["sha256"]:
        raise ValueError("The recorded v28 components or the uploaded v28 file changed.")
    report = {"created_at_utc": utc_now(), "version": args.version, "selected_candidate": name, "candidate": candidate, "learners": {}, "ensembles": {}}

    # 1. the new context networks
    records = {}
    for ensemble in decl["new_networks"]:
        net = decl["networks"][ensemble]
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
    if len(caches) != 1 or len(extras) != 1 or caches != {incumbent["adsb_ranking_table_sha256"]}:
        raise ValueError("The new context seeds used different ADS-B ranking tables (or not v28's) or extra features.")
    report["adsb_ranking_table_sha256"] = caches.pop()
    report["ranking_tokens_content_sha256"] = content_hash(ROOT / "data/ctx/tokens_v1" / "ranking.npz")
    if any(report["ranking_tokens_content_sha256"] != decl["networks"][e]["tokens_manifest_content"]["ranking"] for e in decl["new_networks"]):
        raise ValueError("The ranking token file is not the one the declaration records.")
    frame = ranking_sample(ids, args.sample, 20260927, list(extras.pop()), ROOT / args.adsb_cache, report["adsb_ranking_table_sha256"])
    store = TokenStore(ROOT / "data/ctx/tokens_v1", ["ranking"])
    seeds_by = {}
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
        seeds_by[ensemble] = seeds
    worst = max(r["max_abs_difference_seconds"] for r in report["learners"].values())
    report["reload_tolerance_seconds"], report["reload_max_abs_difference_seconds"] = args.tolerance, worst
    if worst > args.tolerance:
        raise ValueError(f"A reloaded network differs from its saved predictions by {worst} s (tolerance {args.tolerance} s).")
    rnn = read_json(ROOT / "configs/adsb_rnn_v1.json")
    round_a = read_json(ROOT / "configs/adsb_ctx_v2.json")
    learner_lists = {**{e: s["learners"] for e, s in rnn["ensembles"].items()}, **{e: n["learners"] for e, n in round_a["networks"].items()}}
    for ensemble, leaves in learner_lists.items():
        seeds_by[ensemble] = [pd.read_parquet(ROOT / leaf / "refit12" / "ranking_predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
                              for leaf in leaves]
    means = {}
    for ensemble, columns in seeds_by.items():
        if ensemble not in detail:
            raise ValueError(f"The export detail has no {ensemble} column.")
        means[ensemble] = np.mean(columns, axis=0)
        report["ensembles"][ensemble] = {"seeds": len(columns), "max_abs_difference_to_detail": float(np.max(np.abs(means[ensemble] - detail[ensemble].to_numpy(dtype=float))))}
        if report["ensembles"][ensemble]["max_abs_difference_to_detail"] > 1e-9:
            raise ValueError(f"The export's {ensemble} column is not the plain mean of its saved seeds.")

    # 2. components
    recorded = pd.read_parquet(ROOT / incumbent["ranking_values"]).set_index(ID).reindex(ids)
    for column in ["v9", "residual", "committee", "committee_mix", "scope"]:
        if not np.array_equal(detail[column].to_numpy(), recorded[column].to_numpy(), equal_nan=column != "scope"):
            raise ValueError(f"The export's {column} column differs from v28's.")
    report["rebuilt_v28_max_abs_difference_to_recorded"] = float(np.max(np.abs(detail["v28"].to_numpy(dtype=float) - recorded["v28"].to_numpy(dtype=float))))
    if report["rebuilt_v28_max_abs_difference_to_recorded"] > 1e-6:
        raise ValueError("The export's rebuilt v28 differs from the recorded v28.")
    tree_name = rnn["lirf_table"]
    tree_file = ROOT / rnn["extra_feature_tables"][tree_name]["ranking"]
    tree = pd.read_parquet(tree_file).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
    if not np.array_equal(detail[tree_name].to_numpy(dtype=float), tree, equal_nan=True):
        raise ValueError("The export's LIRF tree table differs from the twelve-month LIRF table.")

    # 3. plain arithmetic
    scope = detail["scope"].to_numpy()
    v9 = detail["v9"].to_numpy(dtype=float)
    uploaded = pd.read_parquet(ROOT / incumbent["output"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    out = pq.read_table(ROOT / export["output"]).to_pandas().set_index(ID)[TARGET].reindex(ids).to_numpy()

    def plain(lirf: dict, network: dict) -> np.ndarray:
        net = np.sum([w * means[e] for e, w in network.items()], axis=0)
        values = np.full(len(ids), np.nan)
        for s in LIRF_SCOPES:
            w = lirf[s]
            rows = scope == s
            mix = (w["tree"] * tree[rows] + w["net"] * net[rows]) if w["net"] > 0 else (w["tree"] + w["net"]) * tree[rows]
            values[rows] = np.where(np.isfinite(mix), w["v9"] * v9[rows] + mix, v9[rows])
        return np.maximum(values, -12.0)

    inside = np.isin(scope, LIRF_SCOPES)
    v25 = rnn["v25_recipe"]
    old = plain(v25["lirf"], v25["lirf_network"])
    new = plain(candidate["lirf"], candidate["lirf_network"])
    report["v28_lirf_scopes_max_abs_difference"] = float(np.max(np.abs(old[inside] - recorded["v28"].to_numpy(dtype=float)[inside])))
    report["candidate_lirf_scopes_max_abs_difference_to_detail"] = float(np.max(np.abs(new[inside] - detail[f"v{args.version}"].to_numpy(dtype=float)[inside])))
    report["lirf_scope_rows_equal_after_rounding"] = bool(np.array_equal(np.rint(new[inside]).astype(np.int32), out[inside]))
    report["other_rows_equal_uploaded_v28"] = bool(np.array_equal(out[~inside], uploaded[~inside]))
    report["rows"] = {"lirf_scopes": int(inside.sum()), "other": int((~inside).sum()), "changed_vs_v28": int((out != uploaded).sum())}
    if (report["v28_lirf_scopes_max_abs_difference"] > 1e-6 or report["candidate_lirf_scopes_max_abs_difference_to_detail"] > 1e-6
            or not report["lirf_scope_rows_equal_after_rounding"] or not report["other_rows_equal_uploaded_v28"]):
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
    print({k: report[k] for k in ["passed", "reload_max_abs_difference_seconds", "v28_lirf_scopes_max_abs_difference", "lirf_scope_rows_equal_after_rounding",
                                  "other_rows_equal_uploaded_v28", "rows"]})


if __name__ == "__main__":
    main()
