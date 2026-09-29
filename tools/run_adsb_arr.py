"""Arrival second task: fit, monitor and predict one (variant, seed) of the two-task committee network (configs/adsb_arr_design_v1.json).

The departure side is tools/run_adsb_nn.py's, imported: the same contract (314 columns), rows (LIRF left out of fitting and
monitoring), preprocessing from the departure fitting rows, base recipe (variant nnb of configs/adsb_nn_network_v1.json) and fixed
epoch budget. Added: the arrival rows of tools/build_arr_rows.py (role-swapped, LIRF left out), a second head for the arrival
taxi-in, and the pooled vocabulary. `--no-arrivals` is the nesting check: no arrival row is read and the run is base's.

Fold:
  design    fit the seasonal fold's fit partition (2025-02..06, 08, 09), monitor October, write the October departure predictions
            ONLY (stop_predictions.parquet, column adsb_full). Arrival months: the fitting months, and October unless the variant
            says "arrival_months": "fit". The design never reads a scoring month.

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
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from prc2026_timeline.features import CATEGORICAL_COLUMNS  # noqa: E402
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, TRAINING_MONTHS, attach  # noqa: E402
from run_adsb_nn import build_tables, keep_mask, resolve_variant, rmse  # noqa: E402

FIT_MONTHS = {"design": ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09"]}
TARGET_MONTHS = {"design": ["2025-10"]}
CODE_FILES = ["src/prc2026_arr/__init__.py", "src/prc2026_arr/model.py", "src/prc2026_arr/train.py", "tools/run_adsb_arr.py", "tools/build_arr_rows.py",
              "src/prc2026_nn/__init__.py", "src/prc2026_nn/preprocess.py", "src/prc2026_nn/model.py", "src/prc2026_nn/train.py",
              "tools/run_adsb_nn.py", "tools/run_adsb_learner.py", "src/prc2026_adsb/features.py"]


def code_receipt() -> dict:
    hashes = {p: sha256_file(ROOT / p) for p in CODE_FILES}
    return {"files": hashes, "sha256": object_hash(hashes)}


def arrival_frame(directory: Path, months: list[str], names: list[str], spec: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """The months' arrival rows (declared airports left out, finite taxi-in), the clipped target in seconds, and the input hashes."""
    parts, hashes = [], {}
    for month in months:
        path = directory / f"arr_{month}.parquet"
        record = read_json(directory / f"arr_{month}.json")
        if record["sha256"] != sha256_file(path) or record["month"] != month or record["audit_used"]:
            raise ValueError(f"{path} does not match its record.")
        part = pd.read_parquet(path, columns=[ID, *names, "taxi_in"])
        part["arrival_month"] = month
        parts.append(part)
        hashes[month] = record["sha256"]
    frame = pd.concat(parts, ignore_index=True)
    keep = ~frame["airport"].isin(spec["exclude_airports"]).to_numpy() & np.isfinite(frame["taxi_in"].to_numpy(dtype=float))
    frame = frame.loc[keep].reset_index(drop=True)
    if frame[ID].duplicated().any():
        raise ValueError("Duplicate arrival identities.")
    low, high = spec["clip"]
    return frame, np.clip(frame["taxi_in"].to_numpy(dtype=float), low, high), hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_arr_design_v1.json"))
    parser.add_argument("--variant", help="a declared variant; not with --no-arrivals")
    parser.add_argument("--no-arrivals", action="store_true", help="the nesting check: base's run through this tool")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", required=True, choices=["design"])
    parser.add_argument("--output", type=Path, required=True, help="run directory; the fold is a subdirectory")
    parser.add_argument("--rows", type=Path, default=Path("runs/adsb_arr/rows"))
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--extra-features", type=Path, default=Path("data/external/adsblol/features_v19"))
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_nn/features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke-epochs", type=int, help="smoke only: a shorter budget; the run is marked and no report reads it")
    parser.add_argument("--smoke-rows", type=int, help="smoke only: the first rows of every table")
    args = parser.parse_args()
    if args.no_arrivals == bool(args.variant):
        raise ValueError("Give a declared variant, or --no-arrivals.")
    started = time.time()
    import torch
    torch.set_num_threads(args.threads)
    from prc2026_nn.preprocess import Preprocessor
    from prc2026_nn.train import predict
    from prc2026_arr.train import build_model, pooled_vocabulary, predict_arrival, save, train

    config = read_json(ROOT / args.config)
    if config.get("audit_used"):
        raise ValueError("Configuration does not respect its own contract.")
    variant = resolve_variant(read_json(ROOT / config["base_config"]), config["base_variant"])
    if args.variant and args.variant not in config["variants"]:
        raise ValueError(f"Undeclared variant {args.variant}.")
    task = None if args.no_arrivals else config["variants"][args.variant]
    smoke = bool(args.smoke_epochs or args.smoke_rows)
    if args.smoke_epochs:
        variant["training"]["epochs"] = args.smoke_epochs
    out = ROOT / args.output / args.fold
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    extra_names = list(variant["features"]["extra_features"])
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *extra_names]
    numeric = [n for n in names if n not in CATEGORICAL_COLUMNS]
    categorical = [n for n in names if n in CATEGORICAL_COLUMNS]
    rows_spec = variant["rows"]
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    tables, input_hashes = build_tables(TRAINING_MONTHS, args, extra_names)
    print({"tables": len(tables), "seconds": round(time.time() - started, 1)}, flush=True)

    parts, truths = {}, {}
    assignment = pd.read_parquet(parent / "splits" / "seasonal.parquet")
    for role in ["fit", "early_stop"]:
        features, truth = load_partition(base, parent, catalog, assignment, role, BUNDLE_NAMES)
        if identity_hash(features[ID]) != original["folds"]["seasonal"]["identity_sha256"][role]:
            raise ValueError("Protected split populations changed.")
        parts[role] = attach(features, tables, catalog)
        truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)
        if args.smoke_rows:
            parts[role], truths[role] = parts[role].iloc[:args.smoke_rows].reset_index(drop=True), truths[role][:args.smoke_rows]

    fit_keep = keep_mask(parts["fit"], rows_spec, truths["fit"])
    y_fit = truths["fit"][fit_keep]
    clip = variant["preprocess"].get("target_clip")
    target = np.minimum(y_fit, clip) if clip is not None else y_fit
    fit_frame = parts["fit"].loc[fit_keep].reset_index(drop=True)
    pre = Preprocessor.fit(fit_frame, numeric, categorical, target, bins=variant["preprocess"]["bins"], min_count=variant["preprocess"]["min_count"])
    base_vocab = {k: list(v) for k, v in pre.vocab.items()}

    arrivals, arrival_record, arrival_scaling, held = None, {"used": False}, {}, None
    if task is not None:
        fit_months, target_months = FIT_MONTHS[args.fold], TARGET_MONTHS[args.fold]
        spec = config["arrival"]
        fit_arr, fit_arr_y, hashes = arrival_frame(ROOT / args.rows, fit_months, names, spec)
        tgt_arr, tgt_arr_y, tgt_hashes = arrival_frame(ROOT / args.rows, target_months, names, spec)
        if args.smoke_rows:
            fit_arr, fit_arr_y, tgt_arr, tgt_arr_y = fit_arr.iloc[:args.smoke_rows], fit_arr_y[:args.smoke_rows], tgt_arr.iloc[:args.smoke_rows], tgt_arr_y[:args.smoke_rows]
        mean, scale = float(fit_arr_y.mean()), float(fit_arr_y.std())  # the fitting months' arrivals: the same constants for every variant
        with_target = task["arrival_months"] == "fit_and_target"
        if task["arrival_months"] not in ("fit_and_target", "fit"):
            raise ValueError("Unknown arrival months.")
        arr_frame = pd.concat([fit_arr, tgt_arr], ignore_index=True) if with_target else fit_arr
        arr_y = np.concatenate([fit_arr_y, tgt_arr_y]) if with_target else fit_arr_y
        pre.vocab = pooled_vocabulary(pre, [fit_frame, arr_frame], variant["preprocess"]["min_count"])
        pre.info["categories_kept"] = {k: len(v) for k, v in pre.vocab.items()}
        pre.info["vocabulary"] = "pooled: departure fitting rows and the variant's arrival rows"
        an, ac = pre.transform(arr_frame)
        arrivals = (an, ac, ((arr_y - mean) / scale).astype(np.float32))
        held = (*pre.transform(tgt_arr), tgt_arr_y)
        arrival_scaling = {"mean": mean, "scale": scale, "clip": spec["clip"]}
        arrival_record = {"used": True, "lambda": float(task["lambda"]), "arrival_months": task["arrival_months"], "rows": int(len(arr_y)), "fit_month_rows": int(len(fit_arr_y)),
                          "target_month_rows": int(len(tgt_arr_y)), "target_months_in_training": with_target, "mean": mean, "scale": scale, "clip": spec["clip"],
                          "tables_sha256": {**hashes, **tgt_hashes}, "populated_columns": read_json(ROOT / args.rows / f"arr_{target_months[0]}.json")["populated"],
                          "added_categories": {k: len(set(pre.vocab[k]) - set(base_vocab[k])) for k in categorical}}
        del arr_frame
    x_num, x_cat = pre.transform(fit_frame)
    del fit_frame
    stop_keep = keep_mask(parts["early_stop"], rows_spec, truths["early_stop"])
    stop_frame = parts["early_stop"].loc[stop_keep].reset_index(drop=True)
    monitor = (*pre.transform(stop_frame), truths["early_stop"][stop_keep])
    unseen = np.zeros(len(stop_frame), dtype=bool)  # monitored departures with a category outside base's vocabulary
    for name in ["stand", "operator", "aircraft_type", "runway", "record_flight_prefix"]:
        values = stop_frame[name].astype("string").fillna("__MISSING__")
        unseen |= (~values.isin(base_vocab[name]) & values.ne("__MISSING__")).to_numpy()
    print({"fitted_rows": int(fit_keep.sum()), "monitor_rows": int(stop_keep.sum()), "arrivals": arrival_record.get("rows", 0), "unseen_category_rows": int(unseen.sum()),
           "target_mean": pre.target_mean, "target_scale": pre.target_scale, "seconds": round(time.time() - started, 1)}, flush=True)

    model = build_model(pre, variant["architecture"], args.seed)
    t0 = time.time()
    spec = config["arrival"]
    history = train(model, pre, x_num, x_cat, pre.scale_target(target), variant["training"], args.seed, arrivals=arrivals, lam=arrival_record.get("lambda", 0.0),
                    arrival_batch=spec["batch_size"], generator_offset=spec["generator_offset"], monitor=monitor, log=lambda e: print(e, flush=True))
    fit_seconds = round(time.time() - t0, 1)
    del x_num, x_cat
    save(model, pre, variant["architecture"], arrival_scaling, out)

    values = np.empty(len(parts["early_stop"]))
    for start in range(0, len(values), 200000):
        xn, xc = pre.transform(parts["early_stop"].iloc[start:start + 200000])
        values[start:start + 200000] = predict(model, pre, xn, xc)
    pd.DataFrame({ID: parts["early_stop"][ID].to_numpy(), "adsb_full": values}).to_parquet(out / "stop_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame({ID: stop_frame[ID].to_numpy(), "unseen_category": unseen}).to_parquet(out / "stop_slices.parquet", index=False, compression="zstd")
    monitored = values[stop_keep]
    label = truths["early_stop"][stop_keep]
    record = {"created_at_utc": utc_now(), "fold": args.fold, "stage": "design", "variant": args.variant or "no_arrivals", "task": task, "smoke": smoke, "seed": args.seed,
              "config": args.config.as_posix(), "config_sha256": sha256_file(ROOT / args.config), "base_variant_settings": variant, "base_variant_sha256": object_hash(variant),
              "fitted_rows": int(fit_keep.sum()), "rows": {k: int(len(v)) for k, v in parts.items()}, "epochs": int(variant["training"]["epochs"]), "history": history,
              "fit_seconds": fit_seconds, "preprocessing": pre.info, "target_mean": pre.target_mean, "target_scale": pre.target_scale, "arrival": arrival_record,
              "parameters": int(sum(p.numel() for p in model.parameters())), "stop_rmse_monitored_rows": rmse(label, monitored),
              "stop_rmse_unseen_category_rows": rmse(label[unseen], monitored[unseen]) if unseen.any() else None, "unseen_category_rows": int(unseen.sum()),
              "best_monitor_epoch": int(min(history, key=lambda e: e["monitor_rmse"])["epoch"]), "stop_ids_sha256": identity_hash(parts["early_stop"][ID]),
              "inputs": {"bundle_id": manifest["bundle_id"], **input_hashes, "code": code_receipt(), "features": len(names), "rows": rows_spec, "torch": torch.__version__, "threads": args.threads},
              "model_sha256": sha256_file(out / "model.pt"), "predictions_sha256": sha256_file(out / "stop_predictions.parquet"), "audit_used": False}
    if held is not None:
        taxi_in = predict_arrival(model, held[0], held[1], arrival_scaling["mean"], arrival_scaling["scale"])
        record["arrival"]["target_month_rmse"] = rmse(held[2], taxi_in)
        record["arrival"]["target_month_rmse_of_fit_mean"] = rmse(held[2], np.full(len(taxi_in), arrival_scaling["mean"]))
    record["seconds"] = round(time.time() - started, 1)
    record["environment"] = environment(ROOT)
    write_json(out / "run.json", record)
    print({k: record.get(k) for k in ["fold", "variant", "seed", "smoke", "fitted_rows", "stop_rmse_monitored_rows", "best_monitor_epoch", "fit_seconds", "seconds"]},
          {"arrival": {k: record["arrival"].get(k) for k in ["rows", "lambda", "target_month_rmse", "target_month_rmse_of_fit_mean"]}}, flush=True)


if __name__ == "__main__":
    main()
