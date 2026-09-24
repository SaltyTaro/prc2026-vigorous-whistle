"""Neural committee member on the committee learners' contract: fit, monitor and predict one (variant, seed, fold).

The data path is tools/run_adsb_learner.py's, imported and unchanged: the timeline bundle's 276 model features, the 27 label-free
ADS-B departure/neighbour features from a pass-2 match directory, and (by default) the eleven v19 features, i.e. the 314 columns
of the x19 committee learners. Rows follow the committee: LIRF is left out of fitting and monitoring (every row is predicted).

Folds:
  design    fit the seasonal fold's fit partition; monitor its early-stop partition (October); write the October predictions ONLY.
            The design phase never predicts a scoring month.
  seasonal  the same fit, then also predict the seasonal score partition (January, July): predictions.parquet
  forward   fit January-September, monitor October, predict November: predictions.parquet
  refit     every January-November row at the variant's fixed epoch budget; predict the ranking (ranking_predictions.parquet) and
            December from the opened December build (december_predictions.parquet, also written to the December leaf cache
            runs/december/learners/<output with / as __>.parquet that the development tools read)
  refit12   January-December at the same budget; predict the ranking (refit12/ranking_predictions.parquet)

The schedule is a fixed epoch budget (warm-up + cosine); monitoring never changes training and the saved model is the final
epoch's (EMA) weights, so a refit repeats the development recipe exactly. Variant settings: `defaults` deep-merged with
`variants.<name>` from the JSON config. Output layout matches the GBDT learners (column adsb_full). Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
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
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, RANKING_MONTHS, TRAINING_MONTHS, adsb_table, attach, source_file  # noqa: E402

DECEMBER = "2025-12"
DECEMBER_FILE = source_file(DECEMBER)
CODE_FILES = ["src/prc2026_nn/__init__.py", "src/prc2026_nn/preprocess.py", "src/prc2026_nn/model.py", "src/prc2026_nn/train.py",
              "tools/run_adsb_nn.py", "tools/run_adsb_learner.py", "src/prc2026_adsb/features.py"]


def code_receipt() -> dict:
    hashes = {p: sha256_file(ROOT / p) for p in CODE_FILES}
    return {"files": hashes, "sha256": object_hash(hashes)}


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def resolve_variant(config: dict, name: str) -> dict:
    if name not in config["variants"]:
        raise ValueError(f"Undeclared variant {name}.")
    return merge(config["defaults"], config["variants"][name])


def build_tables(months: list[str], args, extra_names: list[str]) -> tuple[dict, dict]:
    """ADS-B (+ extra) feature tables for every raw file needed, keyed by raw file name; and the input hashes."""
    matches_root = ROOT / args.matches_dir
    stands = pd.read_parquet(matches_root / "stands.parquet")
    cache = ROOT / args.adsb_cache
    jobs = [([m], source_file(m), matches_root, stands, cache) for m in months] + [(RANKING_MONTHS, "ranking.parquet", matches_root, stands, cache)]
    with mp.Pool(min(len(jobs), max(1, args.threads // 2))) as pool:
        built = pool.starmap(adsb_table, jobs)
    tables = {job[1]: table for job, table in zip(jobs, built)}
    hashes = {"stands_sha256": sha256_file(matches_root / "stands.parquet"),
              "matches_sha256": {m: sha256_file(matches_root / f"matches2_{m}.parquet") for m in [*months, *RANKING_MONTHS]},
              "adsb_cache_sha256": {job[1]: sha256_file(cache / f"adsb_{job[1].replace('.parquet', '')}.parquet") for job in jobs}, "extra_feature_tables_sha256": {}}
    if extra_names:
        extra_root = ROOT / args.extra_features
        for ms, raw_name in [([m], source_file(m)) for m in months] + [(RANKING_MONTHS, "ranking.parquet")]:
            files = [extra_root / f"features_v19_{m}.parquet" for m in ms]
            extra = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if [c for c in extra.columns if c != ID] != extra_names:
                raise ValueError(f"Extra feature columns differ in {raw_name}.")
            tables[raw_name] = tables[raw_name].merge(extra, on=ID, how="left", validate="one_to_one")
            hashes["extra_feature_tables_sha256"].update({f.name: sha256_file(f) for f in files})
    return tables, hashes


def keep_mask(frame: pd.DataFrame, rows: dict, labels: np.ndarray | None) -> np.ndarray:
    mask = ~frame["airport"].isin(rows.get("exclude_airports") or []).to_numpy()
    if rows.get("exclude_population"):
        mask &= ~(frame["airport"].eq("LIRF") & frame["nmclock_available"].eq(0) & frame["aircraft_missing"].eq(1)).to_numpy()
    if rows.get("max_label") is not None and labels is not None:
        mask &= labels <= rows["max_label"]
    return mask


def rmse(label: np.ndarray, values: np.ndarray) -> float | None:
    k = np.isfinite(values)
    return float(np.sqrt(np.mean((label[k] - values[k]) ** 2))) if k.any() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", required=True, choices=["design", "seasonal", "forward", "refit", "refit12"])
    parser.add_argument("--output", type=Path, required=True, help="learner directory; the fold is a subdirectory")
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--extra-features", type=Path, default=Path("data/external/adsblol/features_v19"))
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_nn/features"))
    parser.add_argument("--december", type=Path, default=Path("data/december/d198dea10625b0a75977"))
    parser.add_argument("--december-cache", type=Path, default=Path("runs/december/learners"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    import torch
    torch.set_num_threads(args.threads)
    from prc2026_nn.preprocess import Preprocessor
    from prc2026_nn.train import build_model, predict, save, train

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
    rows_spec = variant["rows"]
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    months = [*TRAINING_MONTHS, DECEMBER] if args.fold in ("refit", "refit12") else TRAINING_MONTHS
    tables, input_hashes = build_tables(months, args, extra_names)
    print({"tables": len(tables), "seconds": round(time.time() - started, 1)}, flush=True)

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
    y_fit = truths["fit"][fit_keep]
    clip = variant["preprocess"].get("target_clip")
    target = np.minimum(y_fit, clip) if clip is not None else y_fit
    fit_frame = parts["fit"].loc[fit_keep].reset_index(drop=True)
    pre = Preprocessor.fit(fit_frame, numeric, categorical, target, bins=variant["preprocess"]["bins"], min_count=variant["preprocess"]["min_count"])
    x_num, x_cat = pre.transform(fit_frame)
    del fit_frame
    monitor, stop_keep = None, None
    if "early_stop" in parts:
        stop_keep = keep_mask(parts["early_stop"], rows_spec, truths["early_stop"])
        xs_num, xs_cat = pre.transform(parts["early_stop"].loc[stop_keep].reset_index(drop=True))
        monitor = (xs_num, xs_cat, truths["early_stop"][stop_keep])
    print({"fitted_rows": int(fit_keep.sum()), "numeric": len(numeric), "categorical": len(categorical), "monitor_rows": int(stop_keep.sum()) if stop_keep is not None else 0,
           "target_mean": pre.target_mean, "target_scale": pre.target_scale, "seconds": round(time.time() - started, 1)}, flush=True)

    model = build_model(pre, variant["architecture"], args.seed)
    t0 = time.time()
    history = train(model, pre, x_num, x_cat, pre.scale_target(target), variant["training"], args.seed, monitor=monitor, log=lambda e: print(e, flush=True))
    fit_seconds = round(time.time() - t0, 1)
    del x_num, x_cat
    save(model, pre, variant["architecture"], out)

    def predict_frame(frame: pd.DataFrame) -> np.ndarray:
        values = np.empty(len(frame))
        for start in range(0, len(frame), 200000):
            xn, xc = pre.transform(frame.iloc[start:start + 200000])
            values[start:start + 200000] = predict(model, pre, xn, xc)
        return values

    record = {"created_at_utc": utc_now(), "fold": args.fold, "stage": "design" if args.fold == "design" else ("development" if args.fold in ("seasonal", "forward") else args.fold),
              "variant": args.variant, "variant_settings": variant, "variant_sha256": object_hash(variant), "seed": args.seed, "config": args.config.as_posix(),
              "config_sha256": sha256_file(ROOT / args.config), "fitted_rows": int(fit_keep.sum()), "rows": {k: int(len(v)) for k, v in parts.items()},
              "epochs": int(variant["training"]["epochs"]), "history": history, "fit_seconds": fit_seconds, "preprocessing": pre.info,
              "target_mean": pre.target_mean, "target_scale": pre.target_scale, "parameters": int(sum(p.numel() for p in model.parameters())),
              "inputs": {"bundle_id": manifest["bundle_id"], **input_hashes, **extra_inputs, "code": code_receipt(), "features": len(names), "adsb_features": ADSB_NAMES,
                         "extra_features": extra_names, "rows": rows_spec, "torch": torch.__version__, "threads": args.threads},
              "model_sha256": sha256_file(out / "model.pt"), "audit_used": False}
    if "early_stop" in parts:
        values = predict_frame(parts["early_stop"])
        pd.DataFrame({ID: parts["early_stop"][ID].to_numpy(), "adsb_full": values}).to_parquet(out / "stop_predictions.parquet", index=False, compression="zstd")
        record["stop_rmse_monitored_rows"] = rmse(truths["early_stop"][stop_keep], values[stop_keep])
        record["best_monitor_epoch"] = int(min(history, key=lambda e: e["monitor_rmse"])["epoch"])
        record["stop_ids_sha256"] = identity_hash(parts["early_stop"][ID])
    if "score" in parts:
        values = predict_frame(parts["score"])
        pd.DataFrame({ID: parts["score"][ID].to_numpy(), "adsb_full": values}).to_parquet(out / "predictions.parquet", index=False, compression="zstd")
        score = parts["score"]
        month_of = catalog.set_index(ID)[TIME].reindex(score[ID]).dt.strftime("%Y-%m").to_numpy()
        matched = score["adsb_matched"].eq(1).to_numpy()
        record.update({"score_ids_sha256": identity_hash(score[ID]), "score_rmse": rmse(truths["score"], values),
                       "score_rmse_by_month": {m: rmse(truths["score"][month_of == m], values[month_of == m]) for m in sorted(set(month_of))},
                       "score_rmse_matched": rmse(truths["score"][matched], values[matched]), "score_rmse_unmatched": rmse(truths["score"][~matched], values[~matched])})
    if "december" in parts:
        values = predict_frame(parts["december"])
        table = pd.DataFrame({ID: parts["december"][ID].to_numpy(), "adsb_full": values})
        table.to_parquet(out / "december_predictions.parquet", index=False, compression="zstd")
        cache = ROOT / args.december_cache
        cache.mkdir(parents=True, exist_ok=True)
        leaf = args.output.as_posix().replace("/", "__")
        table.to_parquet(cache / f"{leaf}.parquet", index=False, compression="zstd")
        write_json(cache / f"{leaf}.json", {"kind": "neural", "target": "label", "fold": "refit", "features": len(names), "epochs": record["epochs"],
                                            "fitted_rows": record["fitted_rows"], "model_sha256": record["model_sha256"], "finite_rows": int(np.isfinite(values).sum()),
                                            "source": (args.output / "refit" / "december_predictions.parquet").as_posix()})
        record["december_rows"] = int(len(values))
    if "ranking" in parts:
        values = predict_frame(parts["ranking"])
        pd.DataFrame({ID: parts["ranking"][ID].to_numpy(), "adsb_full": values}).to_parquet(out / "ranking_predictions.parquet", index=False, compression="zstd")
        record.update({"ranking_rows": int(len(values)), "ranking_ids_sha256": identity_hash(parts["ranking"][ID]),
                       "prediction_quantiles": {str(q): float(np.nanquantile(values, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]}})
    record["seconds"] = round(time.time() - started, 1)
    record["environment"] = environment(ROOT)
    write_json(out / "run.json", record)
    print({k: record.get(k) for k in ["fold", "variant", "seed", "fitted_rows", "stop_rmse_monitored_rows", "best_monitor_epoch", "score_rmse_by_month", "fit_seconds", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
