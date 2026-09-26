"""Independent verification of the 32-epoch round's export (configs/adsb_ctx_v4.json on top of v29) against its declaration, selection and
saved networks. Checks, without the export tool's blend code path:
1. every seed of each table new to v29 (ens_ctx_e32; for the combined round also ens_ctx_e24b): its refit12 record carries the declared variant, code, configuration and seed, the saved model hash, the
   declared token content and v29's ADS-B ranking table; the saved model reloads on this CPU and re-predicts a random sample of ranking rows
   (tabular inputs from the exact cached ADS-B table through tools/verify_adsb_nn.ranking_sample, context from the declared ranking token
   file). CPU-trained networks must reproduce within --cpu-tolerance seconds (max abs). GPU-trained ones were trained and predicted on an
   L40S with TF32 matrix products, so a float32 CPU reload cannot match bit for bit:
   each must be within --rms-tolerance seconds RMS and --p99-tolerance seconds at the 99th percentile of its saved ranking predictions (the
   coverage check measured 0.09-0.31 s RMS per network; a wrong or damaged model differs by tens of seconds);
2. every ensemble column of the export detail is the plain mean of its saved seeds (the new tables and the inherited ensembles);
3. the detail's v9, residual, committee, scope, v29 committee mix and round-B LIRF blend columns equal v29's recorded components, and its
   v29 column the recorded v29 (1e-6 s);
4. plain arithmetic on non-LIRF rows, value = max(0.3 x (v9 + residual) + 0.7 x ((1 - s) x committee + s x sum_e w_e x ens_e), -12): with
   pool_s75's weights it must give the uploaded v29 integers, with the selected candidate's the exported integers and the detail (1e-6 s);
   every LIRF row equals the uploaded v29;
5. the submission file has the template's IDs, order and types, and its SHA-256 equals the receipt.
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
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_ctx_v4.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_ctx/dev4/selection.json"))
    parser.add_argument("--export", type=Path, default=Path("docs/adsb_ctx3/export.json"))
    parser.add_argument("--incumbent", type=Path, default=Path("docs/adsb_ctx_lirf/export.json"), help="v29's export record")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--rms-tolerance", type=float, default=0.5)
    parser.add_argument("--p99-tolerance", type=float, default=2.0)
    parser.add_argument("--cpu-tolerance", type=float, default=0.01, help="seconds, max abs, CPU-trained networks")
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_ctx2/inputs/adsb_features_arcw0"), help="a byte copy of arcw0's runs/adsb_nn/features")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    from prc2026_ctx.tokens import TokenStore
    from prc2026_ctx3.train import Context, Store, load, predict
    from prc2026_ctx.train import Context as Context1, load as load1, predict as predict1
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
        raise ValueError("The recorded v29 components or the uploaded v29 file changed.")
    report = {"created_at_utc": utc_now(), "version": args.version, "selected_candidate": name, "candidate": candidate, "learners": {}, "ensembles": {}}

    # 1. the new context networks (tables absent from v29): GPU-trained (record device cuda) or CPU-trained
    v29_tables = {"ens_ctx", "ens_ctx_e24"}
    new_tables = [e for e in decl["networks"] if e not in v29_tables]
    records = {e: {leaf: read_json(ROOT / leaf / "refit12" / "run.json") for leaf in decl["networks"][e]["learners"]} for e in new_tables}
    for ensemble, recs in records.items():
        net = decl["networks"][ensemble]
        for leaf, record in recs.items():
            run = ROOT / leaf / "refit12"
            if (record["fold"] != "refit12" or record["variant_sha256"] != net["variant_sha256"] or record["inputs"]["code"]["sha256"] != net["code"]["sha256"]
                    or record["config_sha256"] != net["config_sha256"] or record["seed"] not in net["seeds"] or record["model_sha256"] != sha256_file(run / "model.pt")
                    or record.get("audit_used") or record["inputs"]["tokens"]["files"] != net["tokens_manifest_content"]):
                raise ValueError(f"{leaf}: refit12 record does not match the declaration.")
        if sorted(r["seed"] for r in recs.values()) != sorted(net["seeds"]):
            raise ValueError(f"The {ensemble} refits do not cover the declared seeds exactly once.")
    every = [r for recs in records.values() for r in recs.values()]
    caches = {r["inputs"]["adsb_cache_sha256"]["ranking.parquet"] for r in every}
    extras = {tuple(r["inputs"]["extra_features"]) for r in every}
    if len(caches) != 1 or len(extras) != 1 or caches != {incumbent["adsb_ranking_table_sha256"]}:
        raise ValueError("The new context seeds used different ADS-B ranking tables (or not v29's) or extra features.")
    report["adsb_ranking_table_sha256"] = caches.pop()
    report["ranking_tokens_content_sha256"] = content_hash(ROOT / "data/ctx/tokens_v1" / "ranking.npz")
    if any(report["ranking_tokens_content_sha256"] != decl["networks"][e]["tokens_manifest_content"]["ranking"] for e in new_tables):
        raise ValueError("The ranking token file is not the one the declaration records.")
    frame = ranking_sample(ids, args.sample, 20260927, list(extras.pop()), ROOT / args.adsb_cache, report["adsb_ranking_table_sha256"])
    host = TokenStore(ROOT / "data/ctx/tokens_v1", ["ranking"])
    device = torch.device("cpu")
    store = Store.build(host, device)
    seeds_by = {}
    for ensemble, recs in records.items():
        seeds_by[ensemble] = []
        for leaf, record in recs.items():
            run = ROOT / leaf / "refit12"
            saved = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["adsb_full"]
            seeds_by[ensemble].append(saved.reindex(ids).to_numpy(dtype=float))
            gpu = record["inputs"].get("device") == "cuda"
            if gpu:
                model, pre, settings = load(run, device)
                x_num, x_cat = pre.transform(frame)
                values = predict(model, pre, torch.from_numpy(x_num), torch.from_numpy(x_cat), Context.build(store, *host.lookup(frame[ID].to_numpy()), settings))
            else:
                model, pre, settings = load1(run)
                x_num, x_cat = pre.transform(frame)
                values = predict1(model, pre, x_num, x_cat, Context1.build(host, *host.lookup(frame[ID].to_numpy()), settings))
            delta = np.abs(values - saved.reindex(frame[ID]).to_numpy(dtype=float))
            report["learners"][leaf] = {"ensemble": ensemble, "seed": record["seed"], "trained_on": "gpu" if gpu else "cpu", "sample_rows": int(len(frame)),
                                        "rms_difference_seconds": float(np.sqrt(np.mean(delta ** 2))), "p99_abs_difference_seconds": float(np.quantile(delta, 0.99)),
                                        "max_abs_difference_seconds": float(delta.max())}
            print(leaf, report["learners"][leaf], flush=True)
    gpu_runs = [r for r in report["learners"].values() if r["trained_on"] == "gpu"]
    cpu_runs = [r for r in report["learners"].values() if r["trained_on"] == "cpu"]
    report["reload"] = {"gpu": {"rms_tolerance_seconds": args.rms_tolerance, "p99_tolerance_seconds": args.p99_tolerance,
                                "worst_rms_seconds": max((r["rms_difference_seconds"] for r in gpu_runs), default=None),
                                "worst_p99_seconds": max((r["p99_abs_difference_seconds"] for r in gpu_runs), default=None),
                                "note": "GPU (TF32) training and prediction, CPU float32 reload"},
                        "cpu": {"max_abs_tolerance_seconds": args.cpu_tolerance, "worst_max_abs_seconds": max((r["max_abs_difference_seconds"] for r in cpu_runs), default=None),
                                "note": "Linux-saved float32 networks reloaded on Windows differ by up to ~2e-3 s (docs/adsb_nn)"}}
    if gpu_runs and (report["reload"]["gpu"]["worst_rms_seconds"] > args.rms_tolerance or report["reload"]["gpu"]["worst_p99_seconds"] > args.p99_tolerance):
        raise ValueError(f"A reloaded GPU network differs from its saved predictions beyond tolerance: {report['reload']}")
    if cpu_runs and report["reload"]["cpu"]["worst_max_abs_seconds"] > args.cpu_tolerance:
        raise ValueError(f"A reloaded CPU network differs from its saved predictions beyond tolerance: {report['reload']}")

    # 2. every ensemble column is the plain mean of its saved seeds
    rnn = read_json(ROOT / "configs/adsb_rnn_v1.json")
    round_b = read_json(ROOT / "configs/adsb_ctx_lirf_v1.json")
    learner_lists = {**{e: s["learners"] for e, s in rnn["ensembles"].items()}, **{e: n["learners"] for e, n in decl["networks"].items() if e not in new_tables},
                     "ens_ctx_lirf": round_b["networks"]["ens_ctx_lirf"]["learners"]}
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

    # 3. components equal v29's
    recorded = pd.read_parquet(ROOT / incumbent["ranking_values"]).set_index(ID).reindex(ids)
    pairs = [("v9", "v9"), ("residual", "residual"), ("committee", "committee"), ("scope", "scope"), ("committee_mix_v29", "committee_mix")]
    pairs += [(c, c) for c in detail.columns if str(c).startswith("lirf_mix:lctx_b100_up")]
    for mine, theirs in pairs:
        if not np.array_equal(detail[mine].to_numpy(), recorded[theirs].to_numpy(), equal_nan=mine != "scope"):
            raise ValueError(f"The export's {mine} column differs from v29's {theirs}.")
    report["rebuilt_v29_max_abs_difference_to_recorded"] = float(np.max(np.abs(detail["v29"].to_numpy(dtype=float) - recorded["v29"].to_numpy(dtype=float))))
    if report["rebuilt_v29_max_abs_difference_to_recorded"] > 1e-6:
        raise ValueError("The export's rebuilt v29 differs from the recorded v29.")

    # 4. plain arithmetic
    uploaded = pd.read_parquet(ROOT / incumbent["output"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    out = pq.read_table(ROOT / export["output"]).to_pandas().set_index(ID)[TARGET].reindex(ids).to_numpy()
    lirf = detail["airport"].eq("LIRF").to_numpy()
    base = 0.3 * (detail["v9"].to_numpy(dtype=float) + detail["residual"].to_numpy(dtype=float))
    committee = detail["committee"].to_numpy(dtype=float)

    def plain(spec: dict) -> np.ndarray:
        network = np.sum([w * means[e] for e, w in spec["network"].items()], axis=0)
        s = float(spec["network_share"])
        return np.maximum(base + 0.7 * ((1 - s) * committee + s * network), -12.0)

    control = plain(decl["candidates"]["control"])
    new = plain(candidate)
    report["control_non_lirf_equal_uploaded_v29"] = bool(np.array_equal(np.rint(control[~lirf]).astype(np.int32), uploaded[~lirf]))
    report["non_lirf_rows_equal_after_rounding"] = bool(np.array_equal(np.rint(new[~lirf]).astype(np.int32), out[~lirf]))
    report["non_lirf_max_abs_difference_to_detail"] = float(np.max(np.abs(new[~lirf] - detail[f"v{args.version}"].to_numpy(dtype=float)[~lirf])))
    report["lirf_rows_equal_uploaded_v29"] = bool(np.array_equal(out[lirf], uploaded[lirf]))
    report["rows"] = {"non_lirf": int((~lirf).sum()), "lirf": int(lirf.sum()), "changed_vs_v29": int((out != uploaded).sum())}
    if not (report["control_non_lirf_equal_uploaded_v29"] and report["non_lirf_rows_equal_after_rounding"] and report["lirf_rows_equal_uploaded_v29"]) \
            or report["non_lirf_max_abs_difference_to_detail"] > 1e-6:
        raise ValueError(f"The exported values do not follow the declared arithmetic: {report}")

    # 5. file contract
    template = pq.read_table(ROOT / "data/raw/submitting.parquet")
    table = pq.read_table(ROOT / export["output"])
    report["file"] = {"sha256": sha256_file(ROOT / export["output"]), "receipt_sha256": export["receipt"]["sha256"], "rows": table.num_rows,
                      "ids_equal_template_order": bool(np.array_equal(table[ID].to_numpy(), template[ID].to_numpy())), "types": {f.name: str(f.type) for f in table.schema}}
    if report["file"]["sha256"] != report["file"]["receipt_sha256"] or not report["file"]["ids_equal_template_order"] or report["file"]["types"] != {ID: "double", TARGET: "int32"}:
        raise ValueError("The submission file breaks the contract.")
    report["passed"] = True
    report["environment"] = environment(ROOT)
    write_json(ROOT / args.output, report)
    print({k: report[k] for k in ["passed", "reload", "control_non_lirf_equal_uploaded_v29", "non_lirf_rows_equal_after_rounding", "lirf_rows_equal_uploaded_v29", "rows"]})


if __name__ == "__main__":
    main()
