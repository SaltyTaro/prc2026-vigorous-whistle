"""Time-gap test of the context network (27 September 2026): tools/run_adsb_ctx3.py with two extra folds and nothing else changed.
  gap    fit January-June 2025 at the variant's fixed epoch budget (no monitor); predict November and December 2025 (5-6 months later)
  near   fit May-October 2025 (the same six-month size, ending just before the scored months); predict November and December 2025
The frozen round-3 folds (design, seasonal, forward, refit, refit12) are unchanged. The question: does a longer-trained network's
development advantage shrink when the scored months lie further from the fitted ones (the ranking months are 1-18 months after the
twelve-month refit, the development folds 0-1 month)? Nothing is uploaded or exported.
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
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from prc2026_timeline.features import CATEGORICAL_COLUMNS  # noqa: E402
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, TRAINING_MONTHS, attach, source_file  # noqa: E402
from run_adsb_nn import DECEMBER, DECEMBER_FILE, build_tables, keep_mask, resolve_variant, rmse  # noqa: E402

CODE_FILES = ["src/prc2026_ctx/__init__.py", "src/prc2026_ctx/tokens.py", "src/prc2026_ctx/model.py", "src/prc2026_ctx/train.py",
              "src/prc2026_ctx2/__init__.py", "src/prc2026_ctx2/tokens.py", "src/prc2026_ctx2/model.py", "src/prc2026_ctx2/train.py", "tools/run_adsb_ctx2.py",
              "src/prc2026_ctx3/__init__.py", "src/prc2026_ctx3/model.py", "src/prc2026_ctx3/train.py", "tools/run_adsb_ctx3.py", "tools/run_adsb_ctx_gap.py",
              "src/prc2026_nn/model.py", "src/prc2026_nn/preprocess.py", "src/prc2026_nn/train.py", "tools/run_adsb_ctx.py", "tools/run_adsb_nn.py",
              "tools/run_adsb_learner.py", "src/prc2026_adsb/features.py", "tools/build_ctx_tokens.py"]
TOKEN_FILES = [*TRAINING_MONTHS, DECEMBER, "ranking"]


def code_receipt() -> dict:
    hashes = {p: sha256_file(ROOT / p) for p in CODE_FILES}
    return {"files": hashes, "sha256": object_hash(hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", required=True, choices=["design", "seasonal", "forward", "refit", "refit12", "gap", "near"])
    parser.add_argument("--output", type=Path, required=True, help="learner directory; the fold is a subdirectory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--extra-features", type=Path, default=Path("data/external/adsblol/features_v19"))
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_nn/features"))
    parser.add_argument("--december", type=Path, default=Path("data/december/d198dea10625b0a75977"))
    parser.add_argument("--december-cache", type=Path, default=Path("runs/december/learners"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit-rows", type=int, default=None, help="smoke test only: subsample the fitting rows (never a result)")
    args = parser.parse_args()
    started = time.time()
    import torch
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    from prc2026_ctx.tokens import TokenStore
    from prc2026_ctx3.train import Context, Store, build_model, predict, save, train
    from prc2026_nn.preprocess import Preprocessor

    config = read_json(ROOT / args.config)
    if config.get("audit_used"):
        raise ValueError("Configuration does not respect its own contract.")
    variant = resolve_variant(config, args.variant)
    out = ROOT / args.output / args.fold
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    extra_names = list(variant["features"]["extra_features"])
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *extra_names]
    numeric = [n for n in names if n not in CATEGORICAL_COLUMNS]
    categorical = [n for n in names if n in CATEGORICAL_COLUMNS]
    rows_spec, context_settings = variant["rows"], variant["context"]
    mask_spec, aux_spec, readouts = variant["masking"], variant["aux"], variant.get("readouts")
    mask_p = float(mask_spec["p"])
    aux_names = list(aux_spec["targets"])
    if mask_p > 0 and any(c not in numeric for c in [*mask_spec["columns"], mask_spec["matched"]]):
        raise ValueError("Masked columns must be numeric model inputs.")
    if aux_names and (mask_p <= 0 or any(c not in mask_spec["columns"] for c in aux_names) or int(variant["architecture"]["n_aux"]) != len(aux_names)):
        raise ValueError("Auxiliary targets need masking, must be masked columns, and n_aux must equal their number.")
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    months = [*TRAINING_MONTHS, DECEMBER] if args.fold in ("refit", "refit12", "gap", "near") else TRAINING_MONTHS
    tables, input_hashes = build_tables(months, args, extra_names)
    tokens_dir = ROOT / variant["tokens_dir"]
    token_manifest = read_json(tokens_dir / "manifest.json")
    host_store = TokenStore(tokens_dir, TOKEN_FILES)
    store = Store.build(host_store, device)
    print({"tables": len(tables), "tokens": int(len(host_store.times)), "device": str(device), "seconds": round(time.time() - started, 1)}, flush=True)

    def december_frame() -> tuple[pd.DataFrame, np.ndarray, dict]:
        meta = read_json(ROOT / args.december / "manifest.json")
        if meta["month"] != DECEMBER or meta["timeline_bundle_id"] != manifest["bundle_id"]:
            raise ValueError("December build does not belong to this timeline bundle.")
        features = pd.read_parquet(ROOT / args.december / "features" / DECEMBER_FILE, columns=[ID, *BUNDLE_NAMES])
        labels = pd.read_parquet(ROOT / args.december / "labels" / DECEMBER_FILE).set_index(ID)[TARGET].reindex(features[ID]).to_numpy(dtype=float)
        if identity_hash(features[ID]) != meta["ids_sha256"] or np.isnan(labels).any():
            raise ValueError("December feature/label identity changed.")
        frame = features.merge(tables[DECEMBER_FILE], on=ID, how="left", validate="one_to_one")
        if frame["adsb_matched"].isna().any():
            raise ValueError("December ADS-B table does not cover every December row.")
        return frame, labels, {"december_build_id": meta["build_id"], "december_ids_sha256": meta["ids_sha256"]}

    def ranking_frame() -> pd.DataFrame:
        ranking = pd.read_parquet(base / "features" / "ranking.parquet", columns=[ID, *BUNDLE_NAMES])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking identities changed.")
        rank = ranking.merge(tables["ranking.parquet"], on=ID, how="left", validate="one_to_one")
        if rank["adsb_matched"].isna().any():
            raise ValueError("ADS-B table does not cover every ranking row.")
        return rank

    parts, truths, extra_inputs = {}, {}, {}
    if args.fold in ("design", "seasonal", "forward"):
        split = "seasonal" if args.fold == "design" else args.fold
        assignment = pd.read_parquet(parent / "splits" / f"{split}.parquet")
        roles = ["fit", "early_stop"] + (["score"] if args.fold != "design" else [])
        for role in roles:
            features, truth = load_partition(base, parent, catalog, assignment, role, BUNDLE_NAMES)
            if identity_hash(features[ID]) != original["folds"][split]["identity_sha256"][role]:
                raise ValueError("Protected split populations changed.")
            parts[role] = attach(features, tables, catalog)
            truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)
    elif args.fold in ("gap", "near"):
        fit_months = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"] if args.fold == "gap" else             ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10"]
        training = catalog[catalog["source_file"].isin([source_file(m) for m in TRAINING_MONTHS])]
        month = training[TIME].dt.strftime("%Y-%m")
        for role, chosen in [("fit", fit_months), ("score", ["2025-11"])]:
            ids = training.loc[month.isin(chosen).to_numpy(), ID].to_numpy()
            assignment = pd.DataFrame({ID: ids, "partition": role})
            features, truth = load_partition(base, parent, catalog, assignment, role, BUNDLE_NAMES)
            parts[role] = attach(features, tables, catalog)
            truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)
        dec, dec_labels, extra_inputs = december_frame()
        parts["score"] = pd.concat([parts["score"], dec], ignore_index=True)
        truths["score"] = np.concatenate([truths["score"], dec_labels])
        extra_inputs = {**extra_inputs, "gap_fit_months": fit_months, "gap_score_months": ["2025-11", DECEMBER]}
        del features
    else:
        training = catalog[catalog["source_file"].isin([source_file(m) for m in TRAINING_MONTHS])]
        if training[TIME].dt.strftime("%Y-%m").eq(DECEMBER).any():
            raise ValueError("The frozen January-November catalog must not contain December.")
        assignment = pd.DataFrame({ID: training[ID].to_numpy(), "partition": "fit"})
        features, truth = load_partition(base, parent, catalog, assignment, "fit", BUNDLE_NAMES)
        frame = attach(features, tables, catalog)
        labels = aligned_truth(frame, truth)[TARGET].to_numpy(dtype=float)
        if len(frame) != 1919370:
            raise ValueError("The January-November refit population changed.")
        dec, dec_labels, extra_inputs = december_frame()
        if args.fold == "refit12":
            frame = pd.concat([frame, dec], ignore_index=True)
            labels = np.concatenate([labels, dec_labels])
            if frame[ID].duplicated().any():
                raise ValueError("Duplicate identities across the twelve months.")
        else:
            parts["december"], truths["december"] = dec, dec_labels
        parts["fit"], truths["fit"] = frame, labels
        parts["ranking"] = ranking_frame()
        del features

    fit_keep = keep_mask(parts["fit"], rows_spec, truths["fit"])
    if args.limit_rows is not None:
        chosen = np.flatnonzero(fit_keep)
        chosen = np.random.default_rng(args.seed).choice(chosen, size=min(args.limit_rows, len(chosen)), replace=False)
        fit_keep = np.zeros(len(fit_keep), dtype=bool)
        fit_keep[chosen] = True
    y_fit = truths["fit"][fit_keep]
    clip = variant["preprocess"].get("target_clip")
    target = np.minimum(y_fit, clip) if clip is not None else y_fit
    fit_frame = parts["fit"].loc[fit_keep].reset_index(drop=True)
    pre = Preprocessor.fit(fit_frame, numeric, categorical, target, bins=variant["preprocess"]["bins"], min_count=variant["preprocess"]["min_count"])
    x_num, x_cat = pre.transform(fit_frame)
    context = Context.build(store, *host_store.lookup(fit_frame[ID].to_numpy()), context_settings)
    x_num, x_cat = torch.from_numpy(x_num).to(device), torch.from_numpy(x_cat).to(device)
    masking, aux, aux_stats = None, None, {}
    if mask_p > 0:
        masking = {"p": mask_p, "columns": torch.tensor([numeric.index(c) for c in mask_spec["columns"]], dtype=torch.long),
                   "matched": numeric.index(mask_spec["matched"])}
    if aux_names:
        matched_fit = fit_frame[mask_spec["matched"]].eq(1).to_numpy()
        columns = []
        for name in aux_names:
            raw = fit_frame[name].to_numpy(dtype=float, na_value=np.nan)
            x = np.where(matched_fit & np.isfinite(raw), np.log1p(np.maximum(raw, 0.0)), np.nan)
            mean, std = float(np.nanmean(x)), float(np.nanstd(x))
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f"Degenerate auxiliary target {name}.")
            columns.append(np.clip((x - mean) / std, -5.0, 5.0))
            aux_stats[name] = {"log1p_mean": mean, "log1p_std": std, "finite_rows": int(np.isfinite(x).sum())}
        aux = {"targets": torch.from_numpy(np.stack(columns, axis=1).astype(np.float32)).to(device), "weight": float(aux_spec["weight"])}
    del fit_frame
    monitor, stop_keep = None, None
    if "early_stop" in parts:
        stop_keep = keep_mask(parts["early_stop"], rows_spec, truths["early_stop"])
        stop_frame = parts["early_stop"].loc[stop_keep].reset_index(drop=True)
        xs_num, xs_cat = pre.transform(stop_frame)
        monitor = (torch.from_numpy(xs_num).to(device), torch.from_numpy(xs_cat).to(device),
                   Context.build(store, *host_store.lookup(stop_frame[ID].to_numpy()), context_settings), truths["early_stop"][stop_keep])
    print({"fitted_rows": int(fit_keep.sum()), "numeric": len(numeric), "categorical": len(categorical), "monitor_rows": int(stop_keep.sum()) if stop_keep is not None else 0,
           "target_mean": pre.target_mean, "target_scale": pre.target_scale, "seconds": round(time.time() - started, 1)}, flush=True)

    model = build_model(pre, variant["architecture"], args.seed, device)
    t0 = time.time()
    history, averaged = train(model, pre, x_num, x_cat, context, torch.from_numpy(pre.scale_target(target)).to(device), variant["training"], args.seed,
                              monitor=monitor, log=lambda e: print(e, flush=True), masking=masking, aux=aux, readouts=readouts)
    fit_seconds = round(time.time() - t0, 1)
    del x_num, x_cat, context, monitor, aux
    if device.type == "cuda":
        torch.cuda.empty_cache()
    save(model, pre, variant["architecture"], context_settings, out, averaged)
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    weights = {"": final_state, **{f"_{name}": state for name, state in sorted(averaged.items())}}

    def predict_frame(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predictions of the final weights (key "") and of every readout (key "_<name>")."""
        result = {}
        for suffix, state in weights.items():
            model.load_state_dict(state)
            result[suffix] = predict_one(frame)
        model.load_state_dict(final_state)
        return result

    def predict_one(frame: pd.DataFrame) -> np.ndarray:
        values = np.empty(len(frame))
        for start in range(0, len(frame), 200000):
            chunk = frame.iloc[start:start + 200000]
            xn, xc = pre.transform(chunk)
            values[start:start + 200000] = predict(model, pre, torch.from_numpy(xn).to(device), torch.from_numpy(xc).to(device),
                                                   Context.build(store, *host_store.lookup(chunk[ID].to_numpy()), context_settings))
        return values

    record = {"created_at_utc": utc_now(), "fold": args.fold, "stage": "design" if args.fold == "design" else ("development" if args.fold in ("seasonal", "forward") else ("time_gap_test" if args.fold in ("gap", "near") else args.fold)),
              "variant": args.variant, "variant_settings": variant, "variant_sha256": object_hash(variant), "seed": args.seed, "config": args.config.as_posix(),
              "config_sha256": sha256_file(ROOT / args.config), "fitted_rows": int(fit_keep.sum()), "rows": {k: int(len(v)) for k, v in parts.items()},
              "epochs": int(variant["training"]["epochs"]), "history": history, "fit_seconds": fit_seconds, "preprocessing": pre.info,
              "target_mean": pre.target_mean, "target_scale": pre.target_scale, "parameters": int(sum(p.numel() for p in model.parameters())),
              "smoke_limit_rows": args.limit_rows,
              "inputs": {"bundle_id": manifest["bundle_id"], **input_hashes, **extra_inputs, "code": code_receipt(), "features": len(names), "adsb_features": ADSB_NAMES,
                         "extra_features": extra_names, "rows": rows_spec, "context": context_settings, "torch": torch.__version__, "threads": args.threads, "device": str(device),
                         "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "float32_matmul_precision": "high",
                         "tokens": {"directory": variant["tokens_dir"], "manifest_sha256": sha256_file(tokens_dir / "manifest.json"), "k": token_manifest["k"],
                                    "files": {n: token_manifest["files"][n]["content_sha256"] for n in TOKEN_FILES}}},
              "model_sha256": sha256_file(out / "model.pt"), "audit_used": False,
              "masking": {"p": mask_p, "columns": list(mask_spec["columns"]) if mask_p > 0 else [], "matched": mask_spec["matched"]},
              "aux": {"targets": aux_names, "weight": float(aux_spec["weight"]), "statistics": aux_stats},
              "readouts": {"settings": readouts, "names": sorted(averaged), "model_sha256": {name: sha256_file(out / f"model_{name}.pt") for name in sorted(averaged)}}}
    if "early_stop" in parts:
        by = predict_frame(parts["early_stop"])
        for suffix, values in by.items():
            pd.DataFrame({ID: parts["early_stop"][ID].to_numpy(), "adsb_full": values}).to_parquet(out / f"stop_predictions{suffix}.parquet", index=False, compression="zstd")
        record["stop_rmse_monitored_rows"] = rmse(truths["early_stop"][stop_keep], by[""][stop_keep])
        record["readout_stop_rmse_monitored_rows"] = {s[1:]: rmse(truths["early_stop"][stop_keep], v[stop_keep]) for s, v in by.items() if s}
        record["best_monitor_epoch"] = int(min(history, key=lambda e: e["monitor_rmse"])["epoch"])
        record["stop_ids_sha256"] = identity_hash(parts["early_stop"][ID])
    if "score" in parts:
        by = predict_frame(parts["score"])
        score = parts["score"]
        month_of = catalog.set_index(ID)[TIME].reindex(score[ID]).dt.strftime("%Y-%m").fillna(DECEMBER).to_numpy()
        matched = score["adsb_matched"].eq(1).to_numpy()
        for suffix, values in by.items():
            pd.DataFrame({ID: score[ID].to_numpy(), "adsb_full": values}).to_parquet(out / f"predictions{suffix}.parquet", index=False, compression="zstd")
        values = by[""]
        record.update({"score_ids_sha256": identity_hash(score[ID]), "score_rmse": rmse(truths["score"], values),
                       "score_rmse_by_month": {m: rmse(truths["score"][month_of == m], values[month_of == m]) for m in sorted(set(month_of))},
                       "score_rmse_matched": rmse(truths["score"][matched], values[matched]), "score_rmse_unmatched": rmse(truths["score"][~matched], values[~matched]),
                       "readout_score_rmse": {s[1:]: rmse(truths["score"], v) for s, v in by.items() if s}})
    if "december" in parts:
        by = predict_frame(parts["december"])
        cache = ROOT / args.december_cache
        cache.mkdir(parents=True, exist_ok=True)
        for suffix, values in by.items():
            table = pd.DataFrame({ID: parts["december"][ID].to_numpy(), "adsb_full": values})
            table.to_parquet(out / f"december_predictions{suffix}.parquet", index=False, compression="zstd")
            leaf = args.output.as_posix().replace("/", "__") + (f"__{suffix[1:]}" if suffix else "")
            table.to_parquet(cache / f"{leaf}.parquet", index=False, compression="zstd")
            write_json(cache / f"{leaf}.json", {"kind": "neural_context3", "target": "label", "fold": "refit", "features": len(names), "epochs": record["epochs"],
                                                "readout": suffix[1:] or "final", "fitted_rows": record["fitted_rows"],
                                                "model_sha256": record["model_sha256"] if not suffix else record["readouts"]["model_sha256"][suffix[1:]],
                                                "finite_rows": int(np.isfinite(values).sum()),
                                                "source": (args.output / "refit" / f"december_predictions{suffix}.parquet").as_posix()})
        record["december_rows"] = int(len(parts["december"]))
    if "ranking" in parts:
        by = predict_frame(parts["ranking"])
        for suffix, values in by.items():
            pd.DataFrame({ID: parts["ranking"][ID].to_numpy(), "adsb_full": values}).to_parquet(out / f"ranking_predictions{suffix}.parquet", index=False, compression="zstd")
        values = by[""]
        record.update({"ranking_rows": int(len(values)), "ranking_ids_sha256": identity_hash(parts["ranking"][ID]),
                       "prediction_quantiles": {str(q): float(np.nanquantile(values, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]}})
    record["seconds"] = round(time.time() - started, 1)
    record["environment"] = environment(ROOT)
    write_json(out / "run.json", record)
    print({k: record.get(k) for k in ["fold", "variant", "seed", "fitted_rows", "stop_rmse_monitored_rows", "best_monitor_epoch", "score_rmse_by_month", "fit_seconds", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
