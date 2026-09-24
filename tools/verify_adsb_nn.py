"""Independent verification of an adsb_nn export (the next versioned submission) against its declaration, selection and saved networks.

Checks, without the export tool's code path for the blend:
1. every declared seed's refit12 record carries the declared variant, code and seed; its saved model reloads (prc2026_nn.train.load) and
   re-predicts a random sample of ranking rows (rebuilt through tools/run_adsb_nn.py's own data path) within --tolerance seconds of its
   saved ranking predictions; the network table in the export detail is the plain mean of the saved seed predictions;
2. the export detail's committee column equals the v23 twelve-month committee table, v9/residual equal v23's recorded components;
3. the candidate value re-derived by plain arithmetic -- on non-LIRF rows 0.3 * (v9 + residual on matched non-LIRF rows) + 0.7 * mix,
   mix = (1 - a) * committee + a * network, then the floor; LIRF rows the uploaded v23 -- equals the exported integers after rounding;
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


def ranking_sample(ids: np.ndarray, sample: int, seed: int, extra_names: list[str], cache: Path, expected_cache_sha256: str) -> pd.DataFrame:
    """Rebuild the network's input frame for a sample of ranking rows through tools/run_adsb_nn.py's data path (bundle + ADS-B + v19).

    The ADS-B table is read from `cache`, which must be byte-identical to the table the networks were fitted and predicted with (its
    SHA-256 is recorded in every run's inputs): a fresh rebuild is not bit-reproducible, because the frozen neighbour features take the
    previous/next departure after an unstable sort by takeoff time, so ties can pick a different neighbour (docs/adsb_nn/README.md)."""
    from prc2026_timeline.bundle import load_bundle
    from run_adsb_learner import BUNDLE_NAMES, RANKING_MONTHS, adsb_table
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    load_bundle(ROOT, base)
    ranking = pd.read_parquet(base / "features" / "ranking.parquet", columns=[ID, *BUNDLE_NAMES])
    pick = np.sort(np.random.default_rng(seed).choice(len(ids), size=min(sample, len(ids)), replace=False))
    ranking = ranking.set_index(ID).reindex(ids[pick]).reset_index()
    if sha256_file(cache / "adsb_ranking.parquet") != expected_cache_sha256:
        raise ValueError("The ADS-B ranking table is not the one the networks used.")
    matches_root = ROOT / "data/external/adsblol/matches"
    table = adsb_table(RANKING_MONTHS, "ranking.parquet", matches_root, pd.read_parquet(matches_root / "stands.parquet"), cache)
    extra = pd.concat([pd.read_parquet(ROOT / "data/external/adsblol/features_v19" / f"features_v19_{m}.parquet") for m in RANKING_MONTHS], ignore_index=True)
    frame = ranking.merge(table, on=ID, how="left", validate="one_to_one").merge(extra, on=ID, how="left", validate="one_to_one")
    if frame["adsb_matched"].isna().any() or [c for c in extra.columns if c != ID] != extra_names:
        raise ValueError("The sample frame is incomplete or the extra features differ.")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_nn_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_nn/dev/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--tolerance", type=float, default=1e-3, help="seconds; a saved network re-predicted on another platform may differ in float rounding")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_nn/features_vm"), help="directory holding the exact adsb_ranking.parquet the networks used")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    from prc2026_nn.train import load, predict
    config = read_json(ROOT / args.config)
    selection = read_json(ROOT / args.selection)
    if selection["config_sha256"] != object_hash(config) or not selection["selected_candidate"]:
        raise ValueError("The selection does not belong to this declaration or selected nothing.")
    name = selection["selected_candidate"]
    share = float(config["candidates"][name])
    export = read_json(ROOT / "docs/adsb_nn/export.json")
    detail = pd.read_parquet(ROOT / export["ranking_values"])
    if sha256_file(ROOT / export["ranking_values"]) != export["ranking_values_sha256"]:
        raise ValueError("The export detail changed after the export.")
    ids = detail[ID].to_numpy()
    report = {"created_at_utc": utc_now(), "version": args.version, "selected_candidate": name, "share": share, "learners": {}}
    # 1. networks
    spec = config["network"]
    recorded_cache = {read_json(ROOT / learner / "refit12" / "run.json")["inputs"]["adsb_cache_sha256"]["ranking.parquet"] for learner in spec["learners"]}
    if len(recorded_cache) != 1:
        raise ValueError("The seeds were fitted with different ADS-B ranking tables.")
    report["adsb_ranking_table_sha256"] = recorded_cache.pop()
    frame = ranking_sample(ids, args.sample, 20260923, spec["variant_settings"]["features"]["extra_features"], ROOT / args.adsb_cache, report["adsb_ranking_table_sha256"])
    seed_predictions = []
    for learner in spec["learners"]:
        run = ROOT / learner / "refit12"
        record = read_json(run / "run.json")
        if record["variant_sha256"] != spec["variant_sha256"] or record["inputs"]["code"]["sha256"] != spec["code_sha256"] or record["seed"] not in spec["seeds"]:
            raise ValueError(f"{learner}: refit12 record does not match the declaration.")
        saved = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["adsb_full"]
        seed_predictions.append(saved.reindex(ids).to_numpy(dtype=float))
        model, pre = load(run)
        x_num, x_cat = pre.transform(frame)
        again = predict(model, pre, x_num, x_cat)
        delta = np.abs(again - saved.reindex(frame[ID]).to_numpy(dtype=float))
        report["learners"][learner] = {"seed": record["seed"], "sample_rows": int(len(frame)), "max_abs_difference_seconds": float(delta.max()),
                                       "rows_above_1e-3_seconds": int((delta > 1e-3).sum()), "model_sha256": sha256_file(run / "model.pt")}
    report["reload_tolerance_seconds"] = args.tolerance
    worst = max(r["max_abs_difference_seconds"] for r in report["learners"].values())
    print({l: r["max_abs_difference_seconds"] for l, r in report["learners"].items()}, flush=True)
    if worst > args.tolerance:
        raise ValueError(f"A reloaded network differs from its saved predictions by {worst} s (tolerance {args.tolerance} s).")
    network = np.mean(seed_predictions, axis=0)
    report["network_table_max_abs_difference"] = float(np.max(np.abs(network - detail["network"].to_numpy(dtype=float))))
    if report["network_table_max_abs_difference"] > 1e-9:
        raise ValueError("The export's network table is not the seed mean.")
    # 2. components
    committee = pd.read_parquet(ROOT / config["extra_feature_tables"][config["committee_table"]]["ranking"]).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
    recorded = pd.read_parquet(ROOT / config["parents"]["v23_ranking_values"]["path"]).set_index(ID).reindex(ids)
    for column, reference in [("committee", committee), ("v9", recorded["v9"].to_numpy(dtype=float)), ("residual", recorded["residual"].to_numpy(dtype=float))]:
        if not np.array_equal(detail[column].to_numpy(dtype=float), reference):
            raise ValueError(f"The export's {column} column differs from its source.")
    # 3. plain-arithmetic candidate
    uploaded = pd.read_parquet(ROOT / config["parents"]["v23_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    airport = detail["airport"].to_numpy()
    lirf = airport == "LIRF"
    base = detail["v9"].to_numpy(dtype=float) + detail["residual"].to_numpy(dtype=float)  # the recorded residual is zero outside its scope
    mix = (1 - share) * committee + share * network
    weight = config["recipe_blend"]["weight"]
    plain = np.maximum((1 - weight) * base + weight * mix, -12.0)
    out = pq.read_table(ROOT / export["output"]).to_pandas().set_index(ID)[TARGET].reindex(ids).to_numpy()
    report["non_lirf_rows_equal_after_rounding"] = bool(np.array_equal(np.rint(plain[~lirf]).astype(np.int32), out[~lirf]))
    report["non_lirf_max_abs_difference_to_detail"] = float(np.max(np.abs(plain[~lirf] - detail[f"v{args.version}"].to_numpy(dtype=float)[~lirf])))
    report["lirf_rows_equal_uploaded_v23"] = bool(np.array_equal(out[lirf], uploaded[lirf]))
    if not (report["non_lirf_rows_equal_after_rounding"] and report["lirf_rows_equal_uploaded_v23"]) or report["non_lirf_max_abs_difference_to_detail"] > 1e-6:
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
    print({k: report[k] for k in ["passed", "network_table_max_abs_difference", "non_lirf_rows_equal_after_rounding", "lirf_rows_equal_uploaded_v23"]},
          {l: r["max_abs_difference_seconds"] for l, r in report["learners"].items()})


if __name__ == "__main__":
    main()
