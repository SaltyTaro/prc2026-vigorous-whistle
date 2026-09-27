"""Leave-one-month-out residual stage on LOMO v9 inputs (the v9 side of the LOMO development set, configs/lomo_dev_v1.json).

The shipped residual stage (v25 adj2 LightGBM + cat CatBoost, v26 rnb network seeds 501-516, combined with rnet_w90) is a 12-month
leave-one-month-out: for a held month it is fitted on the other eleven months' in-scope rows with CROSS-FIT v9 as target base and input.
All three models take v9 as an input feature, so on a LOMO row set whose v9 is the helper's 11-month LOMO v9
(runs/lomo_dev/v9/<month>.parquet) they must be RE-PREDICTED with that v9, as the ranking construction applies the 12-month residual
models to the 12-month refit v9.

  fit --held M [--threads N]     refit adj2 (month weights relative to M) and cat on the eleven other months exactly as
                                 tools/run_adsb_v25.run_month does, save the models under runs/lomo_dev/residual/<M>/, and check that
                                 predicting M with cross-fit v9 reproduces the shipped columns (runs/adsb_v25/dev/predictions_<M>.parquet)
  predict --held M               adj2 / cat from the saved models and the 16 saved rnb held-month networks (runs/adsb_rnn/rnb_s5NN/<M>/),
                                 each first reproduced on cross-fit v9 against its stored predictions, then predicted with LOMO v9 ->
                                 runs/lomo_dev/residual/<M>/predictions.parquet (ID, in_scope, residual:adj2, residual:cat, residual:rnet)
No held-out label is read during fitting.
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
from prc2026.artifacts import environment, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from run_adsb_v11 import scope_mask  # noqa: E402
from run_adsb_v24 import ALL_MONTHS, adsb_features, load_frame, month_table  # noqa: E402

TREE_CONFIG = "configs/adsb_v25_v1.json"
NET_CONFIG = "configs/adsb_rnn_network_v1.json"
SEEDS = list(range(501, 517))
SCOPE = "matched_no_lirf"
OUT = ROOT / "runs/lomo_dev/residual"
CACHE = ROOT / "runs/adsb_v24/features"


def table(config: dict, month: str) -> pd.DataFrame:
    return month_table(load_frame(config, month, "cross_fit"), adsb_features(month, ROOT / config["matches_dirs"]["v11"], CACHE), month)


def with_lomo_v9(t: pd.DataFrame, month: str) -> pd.DataFrame:
    lomo = pd.read_parquet(ROOT / "runs/lomo_dev/v9" / f"{month}.parquet", columns=[ID, "v9"]).set_index(ID)["v9"].reindex(t[ID])
    if lomo.isna().any():
        raise ValueError(f"LOMO v9 does not cover {month}.")
    return t.assign(v9=lomo.to_numpy(dtype=float))


def fit(held: str, threads: int) -> None:
    import lightgbm as lgb  # noqa: F401
    from run_adsb_v25 import fit_variant, predict_variant
    config = read_json(ROOT / TREE_CONFIG)
    out = OUT / held
    if (out / "fit.json").exists():
        raise ValueError(f"Preserve the existing fit: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tables = {m: table(config, m) for m in ALL_MONTHS}
    train = pd.concat([tables[m] for m in ALL_MONTHS if m != held], ignore_index=True)
    shipped = pd.read_parquet(ROOT / f"runs/adsb_v25/dev/predictions_{held}.parquet").set_index(ID).reindex(tables[held][ID])
    record = {"created_at_utc": utc_now(), "held": held, "fit_months": [m for m in ALL_MONTHS if m != held], "threads": threads, "fits": {}}
    for name in ["adj2", "cat"]:
        model = fit_variant(train, config, config["residual_variants"][name], held, threads)
        if model["kind"] == "lightgbm":
            model["model"].save_model(str(out / f"{name}.txt"))
        else:
            model["model"].save_model(str(out / f"{name}.cbm"))
        write_json(out / f"{name}_meta.json", {k: model[k] for k in ["kind", "categories", "rounds", "feature_set", "scope", "rows", "inner_best_iterations", "seconds"]})
        values = predict_variant(model, tables[held], config)
        diff = np.abs(values - shipped[f"residual:{name}"].to_numpy(dtype=float))
        record["fits"][name] = {"rounds": model["rounds"], "rows": model["rows"], "seconds": model["seconds"],
                                "reproduction_vs_shipped_max_abs": float(np.nanmax(diff)), "reproduction_rms": float(np.sqrt(np.nanmean(diff ** 2)))}
        print(held, name, record["fits"][name], flush=True)
    record["seconds"] = round(time.time() - started, 1)
    record["code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_lomo_residual.py", "tools/run_adsb_v25.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]}
    record["environment"] = environment(ROOT)
    write_json(out / "fit.json", record)


def load_tree(held: str, name: str) -> dict:
    meta = read_json(OUT / held / f"{name}_meta.json")
    if meta["kind"] == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(OUT / held / f"{name}.txt"))
    else:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(OUT / held / f"{name}.cbm"))
    return {**meta, "model": model}


def predict(held: str) -> None:
    import torch
    from prc2026_nn.train import load
    from run_adsb_rnn import predict_rows
    from run_adsb_v25 import predict_variant
    torch.set_num_threads(8)
    config = read_json(ROOT / TREE_CONFIG)
    stage = read_json(ROOT / NET_CONFIG)["residual_stage"]
    out = OUT / held
    if (out / "predictions.parquet").exists():
        raise ValueError(f"Preserve the existing predictions: {out}")
    base = table(config, held)
    lomo = with_lomo_v9(base, held)
    shipped = pd.read_parquet(ROOT / f"runs/adsb_v25/dev/predictions_{held}.parquet").set_index(ID).reindex(base[ID])
    inside = np.asarray(scope_mask(base, SCOPE), dtype=bool)
    result = {ID: base[ID].to_numpy(), "in_scope": inside}
    record = {"created_at_utc": utc_now(), "held": held, "lomo_v9": sha256_file(ROOT / "runs/lomo_dev/v9" / f"{held}.parquet"), "checks": {}}
    for name in ["adj2", "cat"]:
        model = load_tree(held, name)
        repro = predict_variant(model, base, config)
        record["checks"][name] = float(np.nanmax(np.abs(repro - shipped[f"residual:{name}"].to_numpy(dtype=float))))
        result[f"residual:{name}"] = predict_variant(model, lomo, config)
    # networks: rows as tools/run_adsb_rnn.month_rows builds them (stage tables, in-scope rows)
    stage_table = month_table(load_frame(stage, held, "cross_fit"), adsb_features(held, ROOT / stage["matches_dirs"]["v11"], CACHE), held)
    if not (stage_table[ID].to_numpy() == base[ID].to_numpy()).all():
        raise ValueError("The residual-stage table and the tree table differ in row order.")
    rows = stage_table.loc[inside].reset_index(drop=True)
    rows_lomo = with_lomo_v9(rows, held)
    nets = []
    for seed in SEEDS:
        directory = ROOT / f"runs/adsb_rnn/rnb_s{seed}/{held}"
        model, pre = load(directory)
        stored = pd.read_parquet(directory / "predictions.parquet").set_index(ID)["residual"].reindex(rows[ID]).to_numpy(dtype=float)
        record["checks"][f"rnb_s{seed}"] = float(np.max(np.abs(predict_rows(model, pre, rows) - stored)))
        column = np.zeros(len(base))
        column[inside] = predict_rows(model, pre, rows_lomo)
        nets.append(column)
    result["residual:rnet"] = np.mean(nets, axis=0)
    # trees must reproduce exactly; the networks' stored predictions were made on the Azure Linux CPUs, so float32 kernels differ slightly
    record["tolerances_s"] = {"trees": 1e-9, "networks": 1e-2}
    record["checks_passed"] = bool(all(v <= (1e-9 if k in ("adj2", "cat") else 1e-2) for k, v in record["checks"].items()))
    if not record["checks_passed"]:
        raise ValueError(f"Reproduction checks failed on {held}: {record['checks']}")
    frame = pd.DataFrame(result)
    for c in frame.columns[2:]:
        if not np.isfinite(frame[c]).all():
            raise ValueError(f"{c} has non-finite values.")
    frame.to_parquet(out / "predictions.parquet", index=False)
    record["predictions_sha256"] = sha256_file(out / "predictions.parquet")
    record["code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_lomo_residual.py", "tools/run_adsb_rnn.py", "tools/run_adsb_v25.py"]}
    write_json(out / "predict.json", record)
    print(held, {k: round(v, 6) for k, v in record["checks"].items()}, "passed" if record["checks_passed"] else "CHECK FAILED", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["fit", "predict"])
    parser.add_argument("--held", required=True, choices=["2025-01", "2025-07"])
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    fit(args.held, args.threads) if args.step == "fit" else predict(args.held)


if __name__ == "__main__":
    main()
