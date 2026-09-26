"""adsb_rctx: a context network (src/prc2026_ctx) as a residual-stage member -- rnet's rows, target and folds with the context view.

The residual stage learns label - v9 (cross-fitted v9, scratch/v24) on matched rows outside LIRF (scope matched_no_lirf); v26-v28 use
0.05 adj2 + 0.05 cat + 0.9 rnet, rnet the 16-seed mean of the tabular network of tools/run_adsb_rnn.py (feature set nb). This tool fits
prc2026_ctx.ContextNet on exactly those rows, target, features (tools/run_adsb_rnn.py's data path, imported unchanged) and folds, and
adds each row's context tokens (tools/build_ctx_tokens.py, --tokens; the committee context network's k = 128 window):

  --held-out M            fit on the in-scope rows of the eleven other 2025 months, predict month M's in-scope rows (zero elsewhere);
                          no held-out label is read during training (no monitor)
  --held-out 2025-10 --design
                          the same fit for October, with October monitored every epoch (the design protocol; October is not a
                          recipe month)
  --refit12               fit on the in-scope rows of all twelve 2025 months, predict the ranking's in-scope rows from the saved
                          label-free ranking table (runs/adsb_rnn/ranking_nb.parquet, tools/refit_adsb_rnn.ranking_rows)

Variant settings: `defaults` deep-merged with `variants.<name>` (architecture/training/context as the committee context network's).
Output: <output>/<variant>_s<seed>/<M | refit12>/{model.pt, preprocessor.*, architecture.json, predictions.parquet or
ranking_predictions.parquet (ID, in_scope, residual), run.json}. Nothing is uploaded.
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
from prc2026.validation import identity_hash  # noqa: E402
from refit_adsb_rnn import ranking_rows  # noqa: E402
from run_adsb_rnn import SCOPE, columns_for, merge, month_rows, resolve_variant, rmse  # noqa: E402
from run_adsb_v24 import ALL_MONTHS  # noqa: E402

DESIGN_MONTH = "2025-10"
TOKEN_FILES = [*ALL_MONTHS, "ranking"]
CODE_FILES = ["src/prc2026_ctx/__init__.py", "src/prc2026_ctx/tokens.py", "src/prc2026_ctx/model.py", "src/prc2026_ctx/train.py",
              "src/prc2026_nn/model.py", "src/prc2026_nn/preprocess.py", "src/prc2026_nn/train.py", "tools/run_adsb_rctx.py", "tools/run_adsb_rnn.py",
              "tools/refit_adsb_rnn.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py", "src/prc2026_adsb/features.py", "tools/build_ctx_tokens.py"]


def code_receipt() -> dict:
    hashes = {p: sha256_file(ROOT / p) for p in CODE_FILES}
    return {"files": hashes, "sha256": object_hash(hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--held-out", choices=ALL_MONTHS)
    parser.add_argument("--refit12", action="store_true")
    parser.add_argument("--design", action="store_true", help="monitor the held-out month every epoch (October only)")
    parser.add_argument("--output", type=Path, required=True, help="learner directory; the held-out month (or refit12) is a subdirectory")
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--tokens", type=Path, default=Path("data/ctx/tokens_v1"))
    parser.add_argument("--ranking-table", type=Path, default=Path("runs/adsb_rnn/ranking_nb.parquet"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="code-path test: 30,000 training rows, one epoch; refuses to write under runs/")
    args = parser.parse_args()
    started = time.time()
    if bool(args.held_out) == args.refit12:
        raise ValueError("Give exactly one of --held-out and --refit12.")
    if args.design and args.held_out != DESIGN_MONTH:
        raise ValueError("The design protocol reads October only.")
    if args.smoke and "runs" in args.output.parts:
        raise ValueError("A smoke test must write outside runs/.")
    import torch
    torch.set_num_threads(args.threads)
    from prc2026_ctx.tokens import TokenStore
    from prc2026_ctx.train import Context, build_model, predict, save, train
    from prc2026_nn.preprocess import Preprocessor

    config = read_json(ROOT / args.config)
    if config.get("audit_used"):
        raise ValueError("Configuration does not respect its own contract.")
    stage = config["residual_stage"]
    if sha256_file(ROOT / stage["declaration"]["path"]) != stage["declaration"]["sha256"]:
        raise ValueError("The residual-stage declaration changed.")
    variant = resolve_variant(config, args.variant)
    if variant["features"]["set"] != "nb":
        raise ValueError("adsb_rctx uses the residual trees' nb columns.")
    if args.smoke:
        variant = merge(variant, {"training": {"epochs": 1}})
    fold = "refit12" if args.refit12 else args.held_out
    out = ROOT / args.output / fold
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    numeric, categorical, _ = columns_for(config, variant)
    cache = ROOT / args.cache
    frames = {m: month_rows(config, m, cache, [], None) for m in ALL_MONTHS}
    fitted_months = ALL_MONTHS if args.refit12 else [m for m in ALL_MONTHS if m != args.held_out]
    train_rows = pd.concat([frames[m][1] for m in fitted_months], ignore_index=True)
    if args.smoke:
        train_rows = train_rows.sample(n=30000, random_state=args.seed).reset_index(drop=True)
    if args.refit12:
        held_all, held_rows = ranking_rows(config, ROOT / args.ranking_table, [], None)
    else:
        held_all, held_rows = frames[args.held_out]
    tokens_dir = ROOT / args.tokens
    token_manifest = read_json(tokens_dir / "manifest.json")
    store = TokenStore(tokens_dir, TOKEN_FILES)
    context_settings = variant["context"]
    print({"fold": fold, "train_rows": len(train_rows), "held_rows": len(held_rows), "numeric": len(numeric), "categorical": len(categorical),
           "tokens": int(len(store.times)), "seconds": round(time.time() - started, 1)}, flush=True)

    target = (train_rows[TARGET] - train_rows["v9"]).to_numpy(dtype=float)
    clip = variant["preprocess"].get("target_clip")
    fit_target = np.clip(target, -clip, clip) if clip is not None else target
    pre = Preprocessor.fit(train_rows, numeric, categorical, fit_target, bins=variant["preprocess"]["bins"], min_count=variant["preprocess"]["min_count"])
    x_num, x_cat = pre.transform(train_rows)
    context = Context.build(store, *store.lookup(train_rows[ID].to_numpy()), context_settings)
    monitor = None
    if args.design:
        m_num, m_cat = pre.transform(held_rows)
        monitor = (m_num, m_cat, Context.build(store, *store.lookup(held_rows[ID].to_numpy()), context_settings),
                   (held_rows[TARGET] - held_rows["v9"]).to_numpy(dtype=float))
    model = build_model(pre, variant["architecture"], args.seed)
    t0 = time.time()
    history = train(model, pre, x_num, x_cat, context, pre.scale_target(fit_target), variant["training"], args.seed, monitor=monitor, log=lambda e: print(e, flush=True))
    fit_seconds = round(time.time() - t0, 1)
    del x_num, x_cat, context
    save(model, pre, variant["architecture"], context_settings, out)

    residual = np.empty(len(held_rows))
    for start in range(0, len(held_rows), 200000):
        chunk = held_rows.iloc[start:start + 200000]
        xn, xc = pre.transform(chunk)
        residual[start:start + 200000] = predict(model, pre, xn, xc, Context.build(store, *store.lookup(chunk[ID].to_numpy()), context_settings))
    column = pd.Series(residual, index=held_rows[ID].to_numpy()).reindex(held_all[ID]).fillna(0.0).to_numpy()
    frame = pd.DataFrame({ID: held_all[ID].to_numpy(), "in_scope": held_all["in_scope"].to_numpy(), "residual": np.where(held_all["in_scope"], column, 0.0)})
    frame.to_parquet(out / ("ranking_predictions.parquet" if args.refit12 else "predictions.parquet"), index=False, compression="zstd")
    record = {"created_at_utc": utc_now(), "stage": "design" if args.design else ("refit12" if args.refit12 else "development"), "fold": fold,
              "held_out": args.held_out, "variant": args.variant, "variant_settings": variant, "variant_sha256": object_hash(variant), "seed": args.seed,
              "config": args.config.as_posix(), "config_sha256": sha256_file(ROOT / args.config), "smoke": args.smoke, "scope": SCOPE,
              "target": "label - cross-fit v9", "fitted_rows": int(len(train_rows)), "fitted_months": fitted_months,
              "fitted_rows_by_month": {m: int(len(frames[m][1])) for m in fitted_months},
              "held_rows": int(len(held_all)), "held_rows_in_scope": int(len(held_rows)), "held_ids_sha256": identity_hash(held_all[ID]),
              "features": {"numeric": numeric, "categorical": categorical}, "epochs": int(variant["training"]["epochs"]), "history": history,
              "fit_seconds": fit_seconds, "preprocessing": pre.info, "target_mean": pre.target_mean, "target_scale": pre.target_scale,
              "parameters": int(sum(p.numel() for p in model.parameters())), "model_sha256": sha256_file(out / "model.pt"),
              "inputs": {"code": code_receipt(), "residual_stage_declaration": stage["declaration"],
                         "scratch": {m: sha256_file(ROOT / stage["cross_fit"]["scratch"] / f"{m}.parquet") for m in ALL_MONTHS},
                         "feature_cache": {m: sha256_file(cache / f"{Path(stage['matches_dirs']['v11']).name}_{m}.parquet") for m in ALL_MONTHS},
                         "context": context_settings, "torch": torch.__version__, "threads": args.threads,
                         "tokens": {"directory": args.tokens.as_posix(), "manifest_sha256": sha256_file(tokens_dir / "manifest.json"), "k": token_manifest["k"],
                                    "files": {n: token_manifest["files"][n]["content_sha256"] for n in TOKEN_FILES}}},
              "audit_used": False}
    if args.refit12:
        record.update({"ranking_table_sha256": sha256_file(ROOT / args.ranking_table), "ranking_rows_in_scope": int(len(held_rows)),
                       "prediction_quantiles": {str(q): float(np.quantile(residual, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]}})
    else:
        label, v9 = held_rows[TARGET].to_numpy(dtype=float), held_rows["v9"].to_numpy(dtype=float)
        record["held_scope_rmse"] = {"v9": rmse(label, v9), "v9_plus_network": rmse(label, v9 + residual)}
    if args.design:
        record["best_monitor_epoch"] = int(min(history, key=lambda e: e["monitor_rmse"])["epoch"])
    record["seconds"] = round(time.time() - started, 1)
    record["environment"] = environment(ROOT)
    write_json(out / "run.json", record)
    print({k: record.get(k) for k in ["fold", "variant", "seed", "fitted_rows", "held_rows_in_scope", "held_scope_rmse", "best_monitor_epoch", "fit_seconds", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
