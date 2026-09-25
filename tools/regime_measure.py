"""coverage_regime_v1 step 2 (configs/coverage_regime_v1.json): the v26 recipe on natural vs thinned ADS-B inputs of one month.

Every ADS-B-dependent member of v26 is re-predicted twice, from the natural and from the thinned feature tables of
tools/regime_features.py, with models that never saw the month: December -> the January-November refit models (the December
leaves), July 2025 -> the seasonal-fold models; residual networks -> their leave-one-month-out models; the residual LightGBM adj2 ->
refitted here on the other eleven months (v25 settings); the CatBoost residual cat -> not refitted (declared approximation: stored
natural value plus adj2's thinned-minus-natural change). The recipe is assembled with tools/run_adsb_rnn_dev.py's own functions.
The natural rebuild must reproduce every stored member prediction and the recorded rnet_w90 RMSE per airport.
Outputs: runs/coverage_regime/measure/<month>/predictions.parquet (every component, natural and thinned) and
docs/coverage_regime/penalty_<month>.json.
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
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026_december.scoring import BUNDLE_NAMES, VocabularyCache, ensemble_prediction, leaf_members, learner_prediction  # noqa: E402
from run_adsb_learner import source_file  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_v11 import candidate_values, rmse, scope_mask  # noqa: E402
from run_adsb_v24 import load_frame, month_table  # noqa: E402

CONFIG = ROOT / "configs/coverage_regime_v1.json"
RECIPE = ROOT / "configs/adsb_rnn_v1.json"
V25 = ROOT / "configs/adsb_v25_v1.json"
DECEMBER_BUILD = ROOT / "data/december/d198dea10625b0a75977"
RUNS = ROOT / "runs/coverage_regime"
DOCS = ROOT / "docs/coverage_regime"
CANDIDATE = "rnet_w90"
CLASSES = ("gate", "dark", "unmatched")


def features(attempt: str, month: str, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(learner frame: bundle + 27 ADS-B + 11 v19 columns, residual feature table) for the month, natural or thinned."""
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    path = DECEMBER_BUILD / "features" / source_file(month) if month == "2025-12" else base / "features" / source_file(month)
    bundle = pd.read_parquet(path, columns=[ID, *BUNDLE_NAMES])
    root = RUNS / attempt / "features" / kind
    committee = pd.read_parquet(root / "committee" / f"adsb_{source_file(month).replace('.parquet', '')}.parquet")
    v19 = pd.read_parquet(root / "v19" / f"features_v19_{month}.parquet")
    frame = bundle.merge(committee, on=ID, how="left", validate="one_to_one").merge(v19, on=ID, how="left", validate="one_to_one")
    if frame["adsb_matched"].isna().any():
        raise ValueError(f"{kind} ADS-B table does not cover every {month} row.")
    residual = pd.read_parquet(root / "residual" / f"matches_v11_{month}.parquet")
    return frame, residual


def stored_member(leaf: str, month: str, ids: np.ndarray) -> np.ndarray:
    if month == "2025-12":
        path = ROOT / "runs/december/learners" / (leaf.replace("/", "__") + ".parquet")
    else:
        path = ROOT / leaf / "seasonal" / "predictions.parquet"
    return pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)


def net_prediction(leaf: str, fold: str, frame: pd.DataFrame, threads: int) -> np.ndarray:
    import torch
    from prc2026_nn.train import load, predict
    torch.set_num_threads(threads)
    model, pre = load(ROOT / leaf / fold)
    values = np.empty(len(frame))
    for start in range(0, len(frame), 100000):
        xn, xc = pre.transform(frame.iloc[start:start + 100000])
        values[start:start + 100000] = predict(model, pre, xn, xc)
    return values


def class_of(table: pd.DataFrame) -> np.ndarray:
    matched = table["adsb_matched"].to_numpy(dtype=float) == 1
    gate = table["adsb_gate"].to_numpy(dtype=float) == 1
    return np.where(~matched, "unmatched", np.where(gate, "gate", "dark"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, choices=["2025-07", "2025-12"])
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    started = time.time()
    threads = min(args.threads, 12)
    declaration = read_json(CONFIG)
    regime = next(v for k, v in declaration["regimes"].items() if k.startswith(args.month))
    attempt = regime["attempt"]
    if sha256_file(RUNS / attempt / f"attempt_{args.month}.json") != regime["attempt_record_sha256"]:
        raise ValueError("The thinning attempt differs from the declared one.")
    recipe = read_json(RECIPE)
    v25_config = read_json(V25)
    candidate = recipe["candidates"][CANDIDATE]
    fold = "refit" if args.month == "2025-12" else "seasonal"
    out = RUNS / "measure" / args.month
    out.mkdir(parents=True, exist_ok=True)
    report = {"created_at_utc": utc_now(), "month": args.month, "attempt": attempt, "declaration_sha256": sha256_file(CONFIG), "recipe_candidate": CANDIDATE,
              "member_fold": fold, "threads": threads, "members": {}}

    frames, residual_features = {}, {}
    for kind in ("natural", "thinned"):
        frames[kind], residual_features[kind] = features(attempt, args.month, kind)
    ids = frames["natural"][ID].to_numpy()
    if not np.array_equal(ids, frames["thinned"][ID].to_numpy()):
        raise ValueError("Natural and thinned frames differ in rows.")

    # 1. blend tables: committee, committee networks, LIRF tree table, LIRF networks -- every leaf re-predicted twice
    vocab = VocabularyCache(ROOT)
    tables = {"natural": {}, "thinned": {}}
    for name, spec in recipe["extra_feature_tables"].items():
        leaves = leaf_members(ROOT, spec["run_dir"])
        preds = {"natural": {}, "thinned": {}}
        for leaf in sorted(set(leaves)):
            is_net = (ROOT / leaf / fold / "model.pt").exists()
            for kind in ("natural", "thinned"):
                if is_net:
                    preds[kind][leaf] = net_prediction(leaf, fold, frames[kind], threads)
                else:
                    preds[kind][leaf], _ = learner_prediction(ROOT, Path(leaf), frames[kind], vocab, threads, fold=fold)
            stored = stored_member(leaf, args.month, ids)
            diff = np.abs(preds["natural"][leaf] - stored)
            report["members"][leaf] = {"table": name, "kind": "network" if is_net else "tree",
                                       "natural_vs_stored_max_abs": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
                                       "nan_pattern_equal": bool(np.array_equal(np.isnan(preds["natural"][leaf]), np.isnan(stored))),
                                       "rows_changed_by_thinning": int((np.abs(preds["thinned"][leaf] - preds["natural"][leaf]) > 1e-9).sum())}
            print(name, leaf, report["members"][leaf], flush=True)
        for kind in ("natural", "thinned"):
            tables[kind][name] = ensemble_prediction(leaves, preds[kind])

    # 2. residual members: rnet (leave-one-month-out networks), adj2 (refit here), cat (approximation)
    from run_adsb_v25 import fit_variant, predict_variant
    frame_v9 = load_frame(recipe, args.month, "cross_fit")
    month_tables = {kind: month_table(frame_v9, residual_features[kind], args.month) for kind in ("natural", "thinned")}
    for kind in ("natural", "thinned"):
        if not np.array_equal(month_tables[kind][ID].to_numpy(), ids):
            order = month_tables[kind].set_index(ID).reindex(ids).reset_index()
            month_tables[kind] = order
    stored_res = pd.read_parquet(ROOT / recipe["residual_columns"][args.month]).set_index(ID).reindex(ids)
    residual = {"natural": {}, "thinned": {}}
    rnet_leaves = recipe["residual_networks"]["rnet"]["learners"]
    for kind in ("natural", "thinned"):
        table = month_tables[kind]
        inside = scope_mask(table, recipe["residual_scope"])
        values = []
        for leaf in rnet_leaves:
            v = np.zeros(len(table))
            v[inside] = net_prediction(leaf, args.month, table.loc[inside].reset_index(drop=True), threads)
            values.append(v)
        residual[kind]["rnet"] = np.mean(values, axis=0)
    stored_rnet = np.mean([pd.read_parquet(ROOT / leaf / args.month / "predictions.parquet").set_index(ID)["residual"].reindex(ids).to_numpy(dtype=float) for leaf in rnet_leaves], axis=0)
    report["residual_check"] = {"rnet_natural_vs_stored_max_abs": float(np.max(np.abs(residual["natural"]["rnet"] - stored_rnet)))}
    other = [m for m in [f"2025-{i:02}" for i in range(1, 13)] if m != args.month]
    from run_adsb_v24 import adsb_features
    cache = ROOT / "runs/adsb_v24/features"
    train = pd.concat([month_table(load_frame(v25_config, m, "cross_fit"), adsb_features(m, ROOT / v25_config["matches_dirs"]["v11"], cache), m) for m in other], ignore_index=True)
    t0 = time.time()
    model = fit_variant(train, v25_config, v25_config["residual_variants"]["adj2"], args.month, threads)
    report["adj2_fit"] = {"rounds": model["rounds"], "inner_best_iterations": model["inner_best_iterations"], "rows": model["rows"], "seconds": round(time.time() - t0, 1)}
    del train
    for kind in ("natural", "thinned"):
        residual[kind]["adj2"] = predict_variant(model, month_tables[kind], v25_config)
    report["residual_check"]["adj2_natural_vs_stored_max_abs"] = float(np.max(np.abs(residual["natural"]["adj2"] - stored_res["residual:adj2"].to_numpy(dtype=float))))
    report["residual_check"]["adj2_natural_vs_stored_rms"] = float(np.sqrt(np.mean((residual["natural"]["adj2"] - stored_res["residual:adj2"].to_numpy(dtype=float)) ** 2)))
    cat_nat = stored_res["residual:cat"].to_numpy(dtype=float)
    in_nat = scope_mask(month_tables["natural"], recipe["residual_scope"])
    in_thin = scope_mask(month_tables["thinned"], recipe["residual_scope"])
    residual["natural"]["cat"] = cat_nat
    residual["thinned"]["cat"] = np.where(in_thin & in_nat, cat_nat + residual["thinned"]["adj2"] - residual["natural"]["adj2"], np.where(in_thin, residual["thinned"]["adj2"], 0.0))
    report["residual_check"]["rows_in_scope"] = {"natural": int(in_nat.sum()), "thinned": int(in_thin.sum()), "thinned_only": int((in_thin & ~in_nat).sum())}

    # 3. the recipe, natural and thinned; attribution by swapping one component group at a time from natural to thinned
    weights = candidate["residual"]
    v25 = recipe["v25_recipe"]
    label = month_tables["natural"][TARGET].to_numpy(dtype=float)
    airports = month_tables["natural"]["airport"].to_numpy()
    floor = float(month_tables["natural"]["floor"].iloc[0])

    def recipe_values(routing: str, blend_kind: str, residual_kind: str) -> np.ndarray:
        table = month_tables[routing].copy()
        for name, values in tables[blend_kind].items():
            table[name] = values
        committee = table[recipe["committee_table"]].to_numpy(dtype=float)
        table["committee_mix:v25"] = mixed(committee, weighted(table, v25["network"]), float(recipe["network_share"]))
        spec = spec_for(recipe, "committee_mix:v25", lirf_blends(recipe, table, v25, "v25"))
        spec = {**spec, "blend_with": {**spec["blend_with"], "weight": float(candidate.get("recipe_weight", recipe["recipe_blend"]["weight"]))}}
        r = np.sum([w * residual[residual_kind][m] for m, w in weights.items()], axis=0)
        if routing == "natural" and residual_kind == "thinned":
            r = np.where(scope_mask(month_tables["natural"], recipe["residual_scope"]), r, 0.0)
        values, _ = candidate_values(table, table["v9"].to_numpy(dtype=float), r, spec, floor)
        return values

    variants = {"natural": recipe_values("natural", "natural", "natural"),
                "blend_tables_thinned": recipe_values("natural", "thinned", "natural"),
                "thinned": recipe_values("thinned", "thinned", "thinned")}
    recorded = read_json(ROOT / f"runs/adsb_rnn/dev/month_{args.month}.json")["candidates"][CANDIDATE]
    natural = variants["natural"]
    report["reproduction"] = {"recorded_rmse": recorded["rmse"], "rebuilt_rmse": rmse(label, natural),
                              "per_airport_max_abs_diff_s": float(max(abs(rmse(label[airports == a], natural[airports == a]) - e["candidate"]) for a, e in recorded["airports"].items()))}
    thinned = variants["thinned"]
    classes_nat, classes_thin = class_of(month_tables["natural"]), class_of(month_tables["thinned"])
    sse = lambda v, k: float(np.sum((label[k] - v[k]) ** 2))  # noqa: E731
    everything = np.ones(len(label), dtype=bool)
    report["whole_month"] = {"rows": int(len(label)), "natural_rmse": rmse(label, natural), "thinned_rmse": rmse(label, thinned),
                             "change_pct": 100 * (rmse(label, thinned) / rmse(label, natural) - 1),
                             "sse_change_blend_tables_only": sse(variants["blend_tables_thinned"], everything) - sse(natural, everything),
                             "sse_change_total": sse(thinned, everything) - sse(natural, everything)}
    non_lirf = airports != "LIRF"
    report["non_lirf"] = {"rows": int(non_lirf.sum()), "natural_rmse": rmse(label[non_lirf], natural[non_lirf]), "thinned_rmse": rmse(label[non_lirf], thinned[non_lirf]),
                          "change_pct": 100 * (rmse(label[non_lirf], thinned[non_lirf]) / rmse(label[non_lirf], natural[non_lirf]) - 1)}
    report["airports"] = {a: {"rows": int((airports == a).sum()), "natural_rmse": rmse(label[airports == a], natural[airports == a]), "thinned_rmse": rmse(label[airports == a], thinned[airports == a]),
                              "change_pct": 100 * (rmse(label[airports == a], thinned[airports == a]) / rmse(label[airports == a], natural[airports == a]) - 1),
                              "rows_changed": int((np.abs(thinned - natural)[airports == a] > 1e-9).sum())} for a in sorted(set(airports))}
    report["transitions"] = {}
    for a in regime["airports_thinned"]:
        k = airports == a
        total_change = sse(thinned, k) - sse(natural, k)
        for cn in CLASSES:
            for ct in CLASSES:
                cell = k & (classes_nat == cn) & (classes_thin == ct)
                if cell.sum() == 0:
                    continue
                report["transitions"][f"{a}:{cn}->{ct}"] = {"rows": int(cell.sum()), "natural_rmse": rmse(label[cell], natural[cell]), "thinned_rmse": rmse(label[cell], thinned[cell]),
                                                            "sse_change": sse(thinned, cell) - sse(natural, cell),
                                                            "share_of_airport_sse_change": (sse(thinned, cell) - sse(natural, cell)) / total_change if total_change else None}
    frame = pd.DataFrame({ID: ids, TARGET: label, "airport": airports, "class_natural": classes_nat, "class_thinned": classes_thin,
                          **{f"recipe:{k}": v for k, v in variants.items()},
                          **{f"{kind}:{name}": v for kind in tables for name, v in tables[kind].items()},
                          **{f"{kind}:residual:{m}": v for kind in residual for m, v in residual[kind].items()}})
    frame.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    report["outputs"] = {"predictions": (out / "predictions.parquet").relative_to(ROOT).as_posix(), "predictions_sha256": sha256_file(out / "predictions.parquet")}
    report["code_sha256"] = {p: sha256_file(ROOT / p) for p in ["tools/regime_measure.py", "tools/regime_features.py", "src/prc2026_regime/thinning.py"]}
    report["seconds"] = round(time.time() - started, 1)
    DOCS.mkdir(parents=True, exist_ok=True)
    write_json(DOCS / f"penalty_{args.month}.json", report)
    print({"reproduction": report["reproduction"], "residual_check": report["residual_check"], "whole_month": report["whole_month"], "non_lirf": report["non_lirf"]}, flush=True)
    print({a: round(e["change_pct"], 3) for a, e in report["airports"].items()}, flush=True)


if __name__ == "__main__":
    main()
