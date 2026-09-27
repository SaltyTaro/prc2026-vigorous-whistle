"""Per-airport fine-tuning test of the 32-epoch context network on the October design fold (27 September 2026).

Starting from each GPU-sprint design network e32_s<seed> (configs/adsb_ctx3_design_v1.json, fitted on the seasonal fold's fit partition
2025-02..06, 08, 09; its October predictions are the base), continue training on ONE airport's fitting rows only (the same fit
partition, the same row rules) with a short, low-rate schedule, then predict that airport's October rows. Every declared
(airport, setting) of configs/adsb_ctx_ft_design_v1.json is run from a fresh copy of the base weights, in one process per seed, so
the fold data and tokens are loaded once. The base network's preprocessor (fitted on all airports) is kept, so inputs are identical.
Outputs runs/adsb_ctx_ft/e32_s<seed>/<airport>_<setting>/{october.parquet, run.json}. Nothing is uploaded or exported.
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
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from run_adsb_learner import BUNDLE_NAMES, TRAINING_MONTHS, attach  # noqa: E402
from run_adsb_nn import build_tables, keep_mask, rmse  # noqa: E402
from run_adsb_ctx3 import TOKEN_FILES, code_receipt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_ctx_ft_design_v1.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-root", type=Path, default=Path("runs/adsb_ctx3_base"), help="holds e32_s<seed>/design")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_ctx_ft"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--extra-features", type=Path, default=Path("data/external/adsblol/features_v19"))
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_nn/features"))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    started = time.time()
    import torch
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    from prc2026_ctx.tokens import TokenStore
    from prc2026_ctx3.train import Context, Store, load, predict, train

    config = read_json(ROOT / args.config)
    if config.get("audit_used") or not config.get("declared_at_utc"):
        raise ValueError("Undeclared or audit-using configuration.")
    base_dir = ROOT / args.base_root / f"e32_s{args.seed}" / "design"
    base_record = read_json(base_dir / "run.json")
    if base_record["variant"] != "e32" or base_record["model_sha256"] != sha256_file(base_dir / "model.pt"):
        raise ValueError("The base network is not the recorded e32 design run.")
    variant = base_record["variant_settings"]
    extra_names = list(variant["features"]["extra_features"])
    rows_spec, context_settings = variant["rows"], variant["context"]
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    tables, input_hashes = build_tables(TRAINING_MONTHS, args, extra_names)
    tokens_dir = ROOT / variant["tokens_dir"]
    token_manifest = read_json(tokens_dir / "manifest.json")
    host_store = TokenStore(tokens_dir, TOKEN_FILES)
    store = Store.build(host_store, device)
    assignment = pd.read_parquet(parent / "splits" / "seasonal.parquet")
    parts, truths = {}, {}
    for role in ["fit", "early_stop"]:
        features, truth = load_partition(base, parent, catalog, assignment, role, BUNDLE_NAMES)
        if identity_hash(features[ID]) != original["folds"]["seasonal"]["identity_sha256"][role]:
            raise ValueError("Protected split populations changed.")
        parts[role] = attach(features, tables, catalog)
        truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)
    del features
    fit_keep = keep_mask(parts["fit"], rows_spec, truths["fit"])
    airport_of = {role: parts[role]["airport"].to_numpy() if "airport" in parts[role] else catalog.set_index(ID)["ADEP_mvt"].reindex(parts[role][ID]).to_numpy()
                  for role in parts}
    base_october = pd.read_parquet(base_dir / "stop_predictions.parquet").set_index(ID)["adsb_full"]
    print({"loaded_seconds": round(time.time() - started, 1), "fit_rows": int(fit_keep.sum())}, flush=True)
    for airport in config["airports"]:
        fit_rows = fit_keep & (airport_of["fit"] == airport)
        oct_rows = airport_of["early_stop"] == airport
        for setting_name, setting in config["settings"].items():
            out = ROOT / args.output / f"e32_s{args.seed}" / f"{airport}_{setting_name}"
            if (out / "run.json").exists():
                print("skip", out, flush=True)
                continue
            t0 = time.time()
            model, pre, _ = load(base_dir, device)
            frame = parts["fit"].loc[fit_rows].reset_index(drop=True)
            x_num, x_cat = pre.transform(frame)
            context = Context.build(store, *host_store.lookup(frame[ID].to_numpy()), context_settings)
            target = torch.from_numpy(pre.scale_target(truths["fit"][fit_rows])).to(device)
            training = {**variant["training"], **setting}
            history, _ = train(model, pre, torch.from_numpy(x_num).to(device), torch.from_numpy(x_cat).to(device), context, target, training, args.seed,
                               log=lambda e: None)
            october = parts["early_stop"].loc[oct_rows].reset_index(drop=True)
            xo_num, xo_cat = pre.transform(october)
            values = predict(model, pre, torch.from_numpy(xo_num).to(device), torch.from_numpy(xo_cat).to(device),
                             Context.build(store, *host_store.lookup(october[ID].to_numpy()), context_settings))
            y = truths["early_stop"][oct_rows]
            base_values = base_october.reindex(october[ID]).to_numpy(dtype=float)
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({ID: october[ID].to_numpy(), "adsb_full": values, "base": base_values}).to_parquet(out / "october.parquet", index=False, compression="zstd")
            record = {"created_at_utc": utc_now(), "airport": airport, "setting": setting_name, "training": training, "seed": args.seed,
                      "base": {"dir": (args.base_root / f"e32_s{args.seed}" / "design").as_posix(), "model_sha256": base_record["model_sha256"]},
                      "fitted_rows": int(fit_rows.sum()), "october_rows": int(oct_rows.sum()), "history": history,
                      "october_rmse": rmse(y, values), "october_rmse_base": rmse(y, base_values), "fit_seconds": round(time.time() - t0, 1),
                      "config_sha256": sha256_file(ROOT / args.config), "inputs": {"bundle_id": manifest["bundle_id"], **input_hashes, "code": code_receipt(),
                      "this_tool_sha256": sha256_file(ROOT / "tools/run_adsb_ctx_ft.py"), "torch": torch.__version__, "device": str(device),
                      "tokens_manifest_sha256": sha256_file(tokens_dir / "manifest.json")}, "audit_used": False, "environment": environment(ROOT)}
            write_json(out / "run.json", record)
            print({"airport": airport, "setting": setting_name, "october_rmse": round(record["october_rmse"], 3), "base": round(record["october_rmse_base"], 3),
                   "seconds": record["fit_seconds"]}, flush=True)
            del model, x_num, x_cat, context, target
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print({"done_seconds": round(time.time() - started, 1)}, flush=True)


if __name__ == "__main__":
    main()
