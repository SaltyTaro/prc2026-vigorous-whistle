"""Fold-provenance residual stage (27 September 2026): the v26 residual stage (adj2 LightGBM, cat CatBoost, rnb network seeds) refitted on
the SAME fitting months as the committee and the context networks, so that every recipe month compares the two sides of the outer blend on
equal footing (docs/outer_weight/README.md: the leave-one-month-out residual and cross-fit v9 see 9-11 months, the fold models 7-9).

Folds (configs/validation_v1.json; December as the committee's and networks' January-November refit):
  seasonal   fit 2025-02..06, 08, 09   -> score 2025-01, 2025-07
  forward    fit 2025-01..09           -> score 2025-11
  december   fit 2025-01..11           -> score 2025-12

Training rows are the in-scope (matched_no_lirf) rows of the fit months with their cross-fit v9, exactly as the shipped residual stage is
trained. Scored rows use the FOLD v9 (tools/run_adsb_v24.load_frame(..., "fold"): the seasonal / forward fold v9 and the January-November
refit v9 for December), the base the committee and networks are compared with. Model settings are the shipped ones, unchanged:
adj2 / cat from configs/adsb_v25_v1.json (tools/run_adsb_v25.fit_variant, adj2's month weights relative to each scored month),
rnb from configs/adsb_rnn_network_v1.json (tools/run_adsb_rnn.fit_network). No held-out label is read during training.

  --trees FOLD [--threads N]          -> runs/fold_residual/trees/predictions_<month>.parquet (residual:adj2, residual:cat) + fit_<fold>.json
  --net FOLD --seed S [--threads N]   -> runs/fold_residual/rnb_s<S>/<month>/{predictions.parquet, run.json}
  --smoke                             code-path test on 30,000 training rows (one epoch / 60 rounds), writes under --output only if not runs/
Nothing is uploaded.
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
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from run_adsb_v11 import scope_mask  # noqa: E402
from run_adsb_v24 import adsb_features, load_frame, month_table  # noqa: E402

VALIDATION = read_json(ROOT / "configs/validation_v1.json")
FOLDS = {"seasonal": {"fit": VALIDATION["folds"]["seasonal"]["fit_months"], "score": ["2025-01", "2025-07"]},
         "forward": {"fit": VALIDATION["folds"]["forward"]["fit_months"], "score": ["2025-11"]},
         "december": {"fit": [f"2025-{m:02d}" for m in range(1, 12)], "score": ["2025-12"]}}
TREE_CONFIG = "configs/adsb_v25_v1.json"
NET_CONFIG = "configs/adsb_rnn_network_v1.json"
SCOPE = "matched_no_lirf"
CODE = ["tools/run_fold_residual.py", "tools/run_adsb_v25.py", "tools/run_adsb_rnn.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py",
        "src/prc2026_adsb/features.py", "src/prc2026_nn/model.py", "src/prc2026_nn/preprocess.py", "src/prc2026_nn/train.py"]


def tables(config: dict, months: list[str], source: str, cache: Path) -> dict[str, pd.DataFrame]:
    root = ROOT / config["matches_dirs"]["v11"]
    return {m: month_table(load_frame(config, m, source), adsb_features(m, root, cache), m) for m in months}


def run_trees(fold: str, cache: Path, out: Path, threads: int, smoke: bool) -> None:
    from run_adsb_v25 import fit_variant, predict_variant
    config = read_json(ROOT / TREE_CONFIG)
    spec = FOLDS[fold]
    started = time.time()
    train = pd.concat(tables(config, spec["fit"], "cross_fit", cache).values(), ignore_index=True)
    if smoke:
        train = train.sample(n=30000, random_state=config["seed"]).reset_index(drop=True)
        config = {**config, "residual_model": {**config["residual_model"], "rounds_cap": 60},
                  "residual_variants": {n: {**v, "rounds_cap": 60} for n, v in config["residual_variants"].items()}}
    scored = tables(config, spec["score"], "fold", cache)
    record = {"created_at_utc": utc_now(), "fold": fold, "fit_months": spec["fit"], "score_months": spec["score"], "smoke": smoke, "threads": threads,
              "training_v9": "cross_fit", "scoring_v9": "fold", "fits": {}}
    out.mkdir(parents=True, exist_ok=True)
    columns = {m: pd.DataFrame({ID: t[ID].to_numpy()}) for m, t in scored.items()}
    for name in ["adj2", "cat"]:
        variant = config["residual_variants"][name]
        targets = spec["score"] if variant.get("weights") else [spec["score"][0]]  # month weights depend on the scored month; cat has none
        for target in targets:
            model = fit_variant(train, config, variant, target, threads)
            record["fits"][f"{name}:{target}"] = {k: model[k] for k in ["kind", "rounds", "inner_best_iterations", "rows", "feature_set", "weights", "weight_sum", "seconds"]}
            print(fold, name, target, record["fits"][f"{name}:{target}"]["rounds"], record["fits"][f"{name}:{target}"]["seconds"], flush=True)
            for month in (spec["score"] if not variant.get("weights") else [target]):
                t = scored[month]
                columns[month][f"residual:{name}"] = predict_variant(model, t, config)
    for month, frame in columns.items():
        t = scored[month]
        inside = np.asarray(scope_mask(t, SCOPE), dtype=bool)
        label, v9 = t[TARGET].to_numpy(dtype=float), t["v9"].to_numpy(dtype=float)
        record.setdefault("scope_rmse", {})[month] = {"rows_in_scope": int(inside.sum()), "fold_v9": float(np.sqrt(np.mean((label - v9)[inside] ** 2))),
                                                       **{c: float(np.sqrt(np.mean((label - v9 - frame[c].to_numpy())[inside] ** 2))) for c in frame.columns if c != ID}}
        frame.to_parquet(out / f"predictions_{month}.parquet", index=False)
    record["code"] = {p: sha256_file(ROOT / p) for p in CODE}
    record["config_sha256"] = sha256_file(ROOT / TREE_CONFIG)
    record["seconds"] = round(time.time() - started, 1)
    record["environment"] = environment(ROOT)
    write_json(out / f"fit_{fold}.json", record)
    print(record["scope_rmse"], flush=True)


def run_net(fold: str, seed: int, cache: Path, out_root: Path, threads: int, smoke: bool) -> None:
    import torch
    from prc2026_nn.train import save
    from run_adsb_rnn import columns_for, fit_network, merge, predict_rows, resolve_variant
    torch.set_num_threads(threads)
    config = read_json(ROOT / NET_CONFIG)
    stage = config["residual_stage"]
    if sha256_file(ROOT / stage["declaration"]["path"]) != stage["declaration"]["sha256"]:
        raise ValueError("The residual-stage declaration changed.")
    variant = resolve_variant(config, "rnb")
    if smoke:
        variant = merge(variant, {"training": {"epochs": 1}})
    numeric, categorical, extra = columns_for(config, variant)
    if extra:
        raise ValueError("rnb must use the nb feature set.")
    spec = FOLDS[fold]
    for month in spec["score"]:  # preserve check before any fitting (review finding D)
        if (out_root / f"rnb_s{seed}" / month / "run.json").exists():
            raise ValueError(f"Preserve the existing run: {out_root / f'rnb_s{seed}' / month}")
    started = time.time()
    fit = tables(stage, spec["fit"], "cross_fit", cache)
    train_rows = pd.concat([t.loc[np.asarray(scope_mask(t, SCOPE), dtype=bool)] for t in fit.values()], ignore_index=True)
    if smoke:
        train_rows = train_rows.sample(n=30000, random_state=seed).reset_index(drop=True)
    model, pre, history, fit_seconds = fit_network(train_rows, numeric, categorical, variant, seed, None, log=lambda e: print(e, flush=True))
    scored = tables(stage, spec["score"], "fold", cache)
    for month, t in scored.items():
        out = out_root / f"rnb_s{seed}" / month
        if (out / "run.json").exists():
            raise ValueError(f"Preserve the existing run: {out}")
        out.mkdir(parents=True, exist_ok=True)
        inside = np.asarray(scope_mask(t, SCOPE), dtype=bool)
        rows = t.loc[inside].reset_index(drop=True)
        residual = predict_rows(model, pre, rows)
        column = np.zeros(len(t))
        column[inside] = residual
        pd.DataFrame({ID: t[ID].to_numpy(), "in_scope": inside, "residual": column}).to_parquet(out / "predictions.parquet", index=False, compression="zstd")
        save(model, pre, variant["architecture"], out)
        label, v9 = rows[TARGET].to_numpy(dtype=float), rows["v9"].to_numpy(dtype=float)
        write_json(out / "run.json", {"created_at_utc": utc_now(), "fold": fold, "variant": "rnb", "variant_sha256": object_hash(variant), "seed": seed,
                                      "fit_months": spec["fit"], "fitted_rows": int(len(train_rows)), "scored_month": month, "rows_in_scope": int(inside.sum()),
                                      "training_v9": "cross_fit", "scoring_v9": "fold", "epochs": int(variant["training"]["epochs"]), "history": history,
                                      "fit_seconds": fit_seconds, "smoke": smoke, "threads": threads,
                                      "held_scope_rmse": {"fold_v9": float(np.sqrt(np.mean((label - v9) ** 2))), "fold_v9_plus_network": float(np.sqrt(np.mean((label - v9 - residual) ** 2)))},
                                      "config_sha256": sha256_file(ROOT / NET_CONFIG), "code": {p: sha256_file(ROOT / p) for p in CODE},
                                      "seconds": round(time.time() - started, 1), "torch": torch.__version__, "environment": environment(ROOT)})
        print(fold, seed, month, {"fit_seconds": fit_seconds, "scope_rmse_v9": round(float(np.sqrt(np.mean((label - v9) ** 2))), 3),
                                  "scope_rmse_v9_net": round(float(np.sqrt(np.mean((label - v9 - residual) ** 2))), 3)}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trees", choices=list(FOLDS))
    parser.add_argument("--net", choices=list(FOLDS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--output", type=Path, default=Path("runs/fold_residual"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and "runs" in args.output.parts:
        raise ValueError("A smoke test must write outside runs/.")
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if args.trees:
        target = out / "trees"
        if (target / f"fit_{args.trees}.json").exists():
            raise ValueError(f"Preserve the existing tree fit for {args.trees}.")
        run_trees(args.trees, ROOT / args.cache, target, args.threads, args.smoke)
    if args.net:
        if args.seed is None:
            raise ValueError("--net needs --seed.")
        run_net(args.net, args.seed, ROOT / args.cache, out, args.threads, args.smoke)


if __name__ == "__main__":
    main()
